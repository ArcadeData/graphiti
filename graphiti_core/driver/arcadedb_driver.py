"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import copy
import logging
import re
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlparse

import httpx
from neo4j import AsyncGraphDatabase, EagerResult
from neo4j.exceptions import ClientError
from typing_extensions import LiteralString

from graphiti_core.driver.arcadedb.operations.community_edge_ops import (
    ArcadeDBCommunityEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.community_node_ops import (
    ArcadeDBCommunityNodeOperations,
)
from graphiti_core.driver.arcadedb.operations.entity_edge_ops import ArcadeDBEntityEdgeOperations
from graphiti_core.driver.arcadedb.operations.entity_node_ops import ArcadeDBEntityNodeOperations
from graphiti_core.driver.arcadedb.operations.episode_node_ops import ArcadeDBEpisodeNodeOperations
from graphiti_core.driver.arcadedb.operations.episodic_edge_ops import (
    ArcadeDBEpisodicEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.graph_ops import ArcadeDBGraphMaintenanceOperations
from graphiti_core.driver.arcadedb.operations.has_episode_edge_ops import (
    ArcadeDBHasEpisodeEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.next_episode_edge_ops import (
    ArcadeDBNextEpisodeEdgeOperations,
)
from graphiti_core.driver.arcadedb.operations.saga_node_ops import ArcadeDBSagaNodeOperations
from graphiti_core.driver.arcadedb.operations.search_ops import ArcadeDBSearchOperations
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.graph_queries import (
    ArcadeDBVectorQuantization,
    get_fulltext_indices,
    get_range_indices,
    get_vector_indices,
)

ARCADEDB_TYPE_BUCKETS = 16
ARCADEDB_VERTEX_TYPES = ('Entity', 'Episodic', 'Community', 'Saga')
ARCADEDB_EDGE_TYPES = ('RELATES_TO', 'MENTIONS', 'HAS_MEMBER', 'HAS_EPISODE', 'NEXT_EPISODE')
_DATABASE_NAME = re.compile(r'[A-Za-z0-9_-]+')
# Write gates the current task already holds, so a write nested in another write (a
# driver.execute_query inside driver.transaction()) does not wait for itself.
_held_write_gates: ContextVar[frozenset[int]] = ContextVar(
    'arcadedb_held_write_gates', default=frozenset()
)

logger = logging.getLogger(__name__)


class ArcadeDBDriver(GraphDriver):
    """Graph driver for ArcadeDB using the Neo4j Bolt wire protocol.

    ArcadeDB ships the Bolt protocol as a server plugin, allowing standard
    Neo4j async drivers to connect directly. This driver reuses the ``neo4j``
    Python async driver for transport while adapting queries for ArcadeDB's
    OpenCypher and SQL dialect.

    Requires ArcadeDB 26.10.1 or newer (Cypher ``CREATE FULLTEXT INDEX`` /
    ``CREATE VECTOR INDEX``, the 3-argument ``db.index.fulltext.query*`` procedures and
    the filtered ``db.index.vector.queryNodes``); developed against 26.12.1. The Bolt
    plugin must be enlisted at server startup and listens on port 7687 by default::

        -Darcadedb.server.plugins=Bolt:com.arcadedb.bolt.BoltProtocolPlugin
    """

    provider = GraphProvider.ARCADEDB
    default_group_id: str = ''

    #: ArcadeDB's HTTP/SQL endpoint listens on this port by default.
    DEFAULT_HTTP_PORT = 2480

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'graphiti',
        http_uri: str | None = None,
        embedding_dim: int | None = None,
        vector_quantization: ArcadeDBVectorQuantization = 'none',
        use_vector_index: bool = True,
        buckets: int = ARCADEDB_TYPE_BUCKETS,
        write_concurrency: int | None = 1,
    ):
        """Connect to ArcadeDB over Bolt.

        Args:
            uri: Bolt URI, e.g. ``bolt://localhost:7687``.
            user: Username.
            password: Password.
            database: Database name.
            http_uri: Base URI of ArcadeDB's HTTP API, e.g. ``http://localhost:2480``.
                Only used to pre-create types with ``BUCKETS`` (no Cypher form).
                Defaults to the Bolt host on port 2480. If it is unreachable a warning
                is logged and types are created implicitly with the default bucket
                count; nothing else depends on it.
            embedding_dim: Dimension of the node embeddings for the HNSW vector index.
                ``None`` lets the index size itself from the first vector written.
            vector_quantization: ``'none'``, ``'scalar'`` or ``'binary'`` quantization of
                the vector index (memory vs. recall).
            use_vector_index: ``False`` skips creating the vector indexes and makes node
                and community similarity search score exactly, in the query.
            buckets: Bucket count for the types pre-created over HTTP.
            write_concurrency: Writers admitted at once per database from this process
                (``None`` for no limit). ArcadeDB resolves concurrent transactions that
                touch the same vertices by aborting one, and the Bolt driver's transaction
                retry then waits (1 s at first) before running it again. Concurrent
                ``add_episode`` calls often share vertices, so one writer at a time
                finishes sooner: 8 writers of 25-edge batches measured 2.0-2.6k edges/s
                gated against 0.5-1.9k ungated (one writer: 2.5k). Writers in other
                processes are not covered; the retry still handles those.
        """
        super().__init__()
        self.client = AsyncGraphDatabase.driver(
            uri=uri,
            auth=(user or '', password or ''),
        )
        self._database = database
        self._user = user or ''
        self._password = password or ''
        self._http_uri = http_uri or self._default_http_uri(uri)
        self.embedding_dim = embedding_dim
        self.vector_quantization: ArcadeDBVectorQuantization = vector_quantization
        self.use_vector_index = use_vector_index
        self._buckets = buckets
        # Clones (one per group_id database) provision themselves on first use; the
        # set and lock are shared by every clone of this driver.
        self._auto_provision = False
        self._default_database = database
        self._provisioned: set[str] = {database}
        self._existing: set[str] = set()
        self._provision_lock = asyncio.Lock()
        self._write_concurrency = write_concurrency
        self._write_gates: dict[str, asyncio.Semaphore] = {}

        # Instantiate ArcadeDB operations
        self._entity_node_ops = ArcadeDBEntityNodeOperations()
        self._episode_node_ops = ArcadeDBEpisodeNodeOperations()
        self._community_node_ops = ArcadeDBCommunityNodeOperations()
        self._saga_node_ops = ArcadeDBSagaNodeOperations()
        self._entity_edge_ops = ArcadeDBEntityEdgeOperations()
        self._episodic_edge_ops = ArcadeDBEpisodicEdgeOperations()
        self._community_edge_ops = ArcadeDBCommunityEdgeOperations()
        self._has_episode_edge_ops = ArcadeDBHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = ArcadeDBNextEpisodeEdgeOperations()
        self._search_ops = ArcadeDBSearchOperations()
        self._graph_ops = ArcadeDBGraphMaintenanceOperations()

        self.aoss_client = None

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """ArcadeDB transaction with real commit/rollback semantics via Bolt."""
        await self._ensure_database()
        async with (
            self._write_gate(self._database),
            self.client.session(database=self._database) as session,
        ):
            tx = await session.begin_transaction()
            try:
                yield _ArcadeDBTransaction(tx)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    @asynccontextmanager
    async def _write_gate(self, database: str) -> AsyncIterator[None]:
        """Admit at most ``write_concurrency`` writers per database from this process.

        A write that already holds the gate (a ``driver.execute_query`` inside
        ``driver.transaction()``) passes straight through instead of waiting for itself.
        """
        if self._write_concurrency is None:
            yield
            return
        gate = self._write_gates.get(database)
        if gate is None:
            gate = self._write_gates[database] = asyncio.Semaphore(self._write_concurrency)
        held = _held_write_gates.get()
        if id(gate) in held:
            yield
            return
        async with gate:
            token = _held_write_gates.set(held | {id(gate)})
            try:
                yield
            finally:
                _held_write_gates.reset(token)

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        kwargs.setdefault('database_', self._database)
        if kwargs.get('routing_') == 'r':
            if not await self._readable(kwargs['database_']):
                return EagerResult(records=[], summary=None, keys=[])  # type: ignore[arg-type]
        else:
            await self._ensure_database(kwargs['database_'])

        gate = (
            nullcontext()
            if kwargs.get('routing_') == 'r'
            else self._write_gate(kwargs['database_'])
        )
        try:
            async with gate:
                result = await self.client.execute_query(
                    cypher_query_, parameters_=params, **kwargs
                )
        except Exception as e:
            logger.error(f'Error executing ArcadeDB query: {e}\n{cypher_query_}\n{params}')
            raise

        return result

    def clone(self, database: str) -> 'ArcadeDBDriver':
        """Return a driver bound to another database.

        GraphDriver.clone() returns self, which is wrong here in the one place
        it matters most: on ArcadeDB a database IS the tenant boundary, so
        graphiti's per-tenant switch has to change which database subsequent
        queries reach. Returning self left every call on the previous database,
        which does not raise -- it returns no rows, so a caller sees an empty
        tenant rather than an error.

        The Bolt client is shared: it is not bound to a database (session() and
        execute_query() take database_ per call), so closing one clone closes
        them all, as it does for the other drivers.

        Graphiti's default group_id (``''``) names no database; it means the database
        this driver was constructed with, as for FalkorDB's default graph.
        """
        if database == self.default_group_id:
            database = self._default_database
        cloned = copy.copy(self)
        cloned._database = database
        # Graphiti derives these databases from group_id, so nothing else would ever
        # create them: the clone provisions its database (and indexes) on first use.
        cloned._auto_provision = True
        return cloned

    async def _ensure_database(self, database: str | None = None) -> None:
        """Create ``database`` and its indexes the first time this process uses it.

        Only clones do this (``_auto_provision``); the database a driver was constructed
        with is the caller's to manage. Existence is probed over Bolt, so an existing
        database needs no HTTP access; creating one needs the HTTP API (``http_uri``).
        Indexes are built every time a database is first seen, not only when it was
        created, because they are idempotent and a database made by an earlier run may
        have been abandoned half way.
        """
        database = database or self._database
        if not self._auto_provision or database in self._provisioned:
            return

        async with self._provision_lock:
            if database in self._provisioned:
                return
            if not _DATABASE_NAME.fullmatch(database):
                raise ValueError(f'Invalid ArcadeDB database name: {database!r}')

            if not await self._database_exists(database):
                await self._create_database(database)

            scoped = copy.copy(self)
            scoped._database = database
            scoped._auto_provision = False
            await scoped.build_indices_and_constraints()
            self._provisioned.add(database)

    async def _readable(self, database: str) -> bool:
        """Whether a read may go to ``database``; a missing one is an empty tenant, not an error.

        A search with a mistyped ``group_id`` used to create (and index) a database just to
        find nothing in it. Reads of a missing database now come back empty and leave
        creation to the first write. Existence is cached once seen, so only the first read
        of a tenant pays the probe.
        """
        if not self._auto_provision or database in self._provisioned or database in self._existing:
            return True
        if not _DATABASE_NAME.fullmatch(database):
            raise ValueError(f'Invalid ArcadeDB database name: {database!r}')
        if await self._database_exists(database):
            self._existing.add(database)
            return True
        return False

    async def _database_exists(self, database: str) -> bool:
        try:
            async with self.client.session(database=database) as session:
                result = await session.run('RETURN 1')
                await result.consume()
            return True
        except ClientError as e:
            if e.code == 'Neo.ClientError.Database.DatabaseNotFound':
                return False
            raise

    async def _create_database(self, database: str) -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._http_uri}/api/v1/server',
                    json={'command': f'create database {database}'},
                    auth=(self._user, self._password),
                    timeout=30.0,
                )
            # Another process may have created it between the probe and here.
            if response.status_code >= 400 and 'already exists' not in response.text:
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f'ArcadeDB database {database!r} does not exist and could not be created via '
                f'the HTTP API at {self._http_uri} ({e}). Graphiti uses one database per '
                f'group_id; pass http_uri= or create the database yourself.'
            ) from e

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return _ArcadeDBSession(self, self.client.session(database=_database), _database)

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine:
        # ArcadeDB does not support Neo4j's CALL db.indexes() YIELD name DROP INDEX name.
        # Return a no-op coroutine. Indexes should be managed via ArcadeDB SQL DDL.
        async def _noop():
            logger.warning(
                'delete_all_indexes via Bolt is not supported for ArcadeDB. '
                'Use ArcadeDB SQL DDL to manage indexes.'
            )

        return _noop()

    @classmethod
    def _default_http_uri(cls, bolt_uri: str) -> str:
        """Derive the HTTP API base URI from the Bolt URI, swapping in the HTTP port."""
        parsed = urlparse(bolt_uri)
        host = parsed.hostname or 'localhost'
        scheme = (
            'https' if parsed.scheme in ('bolt+s', 'bolt+ssc', 'neo4j+s', 'neo4j+ssc') else 'http'
        )
        return f'{scheme}://{host}:{cls.DEFAULT_HTTP_PORT}'

    async def _create_types(self) -> None:
        """Pre-create the vertex/edge types with a fixed bucket count over HTTP/SQL.

        ArcadeDB creates types implicitly with a default bucket count; more buckets let
        concurrent writers land on different buckets. ``BUCKETS`` can only be set at
        creation (ALTER TYPE rejects it) and has no Cypher form, so this needs the HTTP
        API. Best-effort: if it is unreachable the types are created implicitly later with
        the default bucket count, which only costs write concurrency.
        """
        statements = [
            f'CREATE VERTEX TYPE {name} IF NOT EXISTS BUCKETS {self._buckets}'
            for name in ARCADEDB_VERTEX_TYPES
        ] + [
            f'CREATE EDGE TYPE {name} IF NOT EXISTS BUCKETS {self._buckets}'
            for name in ARCADEDB_EDGE_TYPES
        ]

        async with httpx.AsyncClient() as client:
            for statement in statements:
                try:
                    response = await client.post(
                        f'{self._http_uri}/api/v1/command/{self._database}',
                        json={'language': 'sql', 'command': statement},
                        auth=(self._user, self._password),
                        timeout=30.0,
                    )
                    response.raise_for_status()
                except httpx.HTTPError as e:
                    logger.warning(
                        f'Could not pre-create ArcadeDB types with BUCKETS {self._buckets} via '
                        f'the HTTP API at {self._http_uri} ({e}); types will be created '
                        f'implicitly with the default bucket count. Pass http_uri= if the HTTP '
                        f'API is not on port {self.DEFAULT_HTTP_PORT}.'
                    )
                    return

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        await self._create_types()

        queries: list[LiteralString] = get_range_indices(self.provider) + get_fulltext_indices(
            self.provider
        )
        if self.use_vector_index:
            queries += get_vector_indices(
                self.provider, self.embedding_dim, self.vector_quantization
            )

        for query in queries:
            try:
                await self.execute_query(query)
            except Exception as e:
                # Every statement is IF NOT EXISTS, so a failure here is real.
                logger.warning(f'ArcadeDB index creation failed: {e}\n{query}')

    async def health_check(self) -> None:
        """Check ArcadeDB connectivity via the Bolt driver."""
        try:
            await self.client.verify_connectivity()
            return None
        except Exception as e:
            print(f'ArcadeDB health check failed: {e}')
            raise


class _ArcadeDBTransaction(Transaction):
    """Wraps a Neo4j AsyncTransaction for ArcadeDB Bolt protocol."""

    def __init__(self, tx: Any):
        self._tx = tx

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._tx.run(query, **kwargs)


class _ArcadeDBSession(GraphDriverSession):
    """Session that admits writes through the driver's write gate and, on an
    auto-provisioning clone, makes sure its database exists first.

    ``ArcadeDBDriver.session()`` is synchronous and callers use the session without
    entering it (``session.execute_write(...)``), so both have to hang off the
    operations themselves. ``run`` and ``begin_transaction`` hand the work back to the
    caller, so they are not gated.
    """

    provider = GraphProvider.ARCADEDB

    def __init__(self, driver: 'ArcadeDBDriver', session: Any, database: str):
        self._driver = driver
        self._session = session
        self._database = database

    async def __aenter__(self):
        await self._driver._ensure_database(self._database)
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._session.__aexit__(exc_type, exc, tb)

    async def run(self, query: str, **kwargs: Any) -> Any:
        await self._driver._ensure_database(self._database)
        return await self._session.run(query, **kwargs)

    async def close(self):
        return await self._session.close()

    async def execute_write(self, func, *args, **kwargs):
        await self._driver._ensure_database(self._database)
        async with self._driver._write_gate(self._database):
            return await self._session.execute_write(func, *args, **kwargs)

    async def execute_read(self, func, *args, **kwargs):
        await self._driver._ensure_database(self._database)
        return await self._session.execute_read(func, *args, **kwargs)

    async def begin_transaction(self, *args, **kwargs):
        await self._driver._ensure_database(self._database)
        return await self._session.begin_transaction(*args, **kwargs)
