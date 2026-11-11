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

Live tests against an ArcadeDB server (build 601600ee or newer) with the Bolt plugin enabled.
Each test gets its own throwaway database, created and dropped over the HTTP API at
ARCADEDB_HTTP_URI (see tests/helpers_test.py).
"""

import asyncio
import contextlib
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.arcadedb_driver import ArcadeDBDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graph_queries import get_vector_cosine_func_query
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import (
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_similarity_search,
    edge_fulltext_search,
    edge_similarity_search,
    node_fulltext_search,
    node_similarity_search,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import ensure_utc
from tests.helpers_test import (
    ARCADEDB_HTTP_URI,
    ARCADEDB_PASSWORD,
    ARCADEDB_URI,
    ARCADEDB_USER,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv('DISABLE_ARCADEDB') is not None, reason='DISABLE_ARCADEDB is set'),
]

DIM = 64
AUTH = (ARCADEDB_USER, ARCADEDB_PASSWORD)


def _admin(command: str) -> None:
    response = httpx.post(
        f'{ARCADEDB_HTTP_URI}/api/v1/server', json={'command': command}, auth=AUTH, timeout=60
    )
    response.raise_for_status()


def _sql(database: str, command: str) -> list[dict]:
    response = httpx.post(
        f'{ARCADEDB_HTTP_URI}/api/v1/command/{database}',
        json={'language': 'sql', 'command': command},
        auth=AUTH,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()['result']


def _new_driver(database: str) -> ArcadeDBDriver:
    return ArcadeDBDriver(
        uri=ARCADEDB_URI,
        user=ARCADEDB_USER,
        password=ARCADEDB_PASSWORD,
        database=database,
        http_uri=ARCADEDB_HTTP_URI,
        embedding_dim=DIM,
    )


@pytest.fixture
async def database():
    name = f'gi_it_{uuid4().hex[:10]}'
    _admin(f'create database {name}')
    try:
        yield name
    finally:
        _admin(f'drop database {name}')


@pytest.fixture
async def driver(database):
    d = _new_driver(database)
    await d.build_indices_and_constraints()
    try:
        yield d
    finally:
        await d.close()


def _unit(
    rng: random.Random, center: list[float] | None = None, noise: float = 0.05
) -> list[float]:
    if center is None:
        v = [rng.gauss(0, 1) for _ in range(DIM)]
    else:
        v = [c + rng.gauss(0, noise) for c in center]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _axis(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIM)]


def _entity(
    name: str, group: str, embedding: list[float], labels: list[str] | None = None, **kwargs
) -> EntityNode:
    return EntityNode(
        name=name,
        group_id=group,
        labels=labels or ['Entity'],
        summary=f'{name} summary',
        name_embedding=embedding,
        **kwargs,
    )


def _record_queries(driver) -> list[str]:
    queries: list[str] = []
    original = driver.execute_query

    async def recording(query, **kwargs):
        queries.append(query)
        return await original(query, **kwargs)

    driver.execute_query = recording
    return queries


async def _both_paths_agree(via_search_utils, via_ops) -> list[str]:
    """The live search path and the ops contract must answer identically."""
    live = [n.uuid for n in await via_search_utils]
    ops = [n.uuid for n in await via_ops]
    assert live == ops
    return live


async def test_ann_is_scoped_to_the_tenant_before_top_k(driver):
    """Tenants B and C hold the nearest vectors. A plain top-k then a group filter would
    return nothing for A; the group filter has to run inside the index search."""
    rng = random.Random(1)
    query = _axis(0)
    nodes = []
    for i in range(30):
        nodes.append(_entity(f'a{i}', 'A', _unit(rng, _axis(1), 0.3)))
        nodes.append(_entity(f'b{i}', 'B', _unit(rng, _axis(0), 0.02)))
        nodes.append(_entity(f'c{i}', 'C', _unit(rng, _axis(0), 0.02)))
    for n in nodes:
        await n.save(driver)

    queries = _record_queries(driver)
    uuids = await _both_paths_agree(
        node_similarity_search(driver, query, SearchFilters(), ['A'], limit=10, min_score=0.0),
        driver.search_ops.node_similarity_search(
            driver, query, SearchFilters(), ['A'], limit=10, min_score=0.0
        ),
    )

    assert any('db.index.vector.queryNodes' in q for q in queries), 'ANN path was not used'
    assert len(uuids) == 10
    a_uuids = {n.uuid for n in nodes if n.group_id == 'A'}
    assert set(uuids) <= a_uuids


async def test_ann_recall_and_score_scale_match_the_exact_scan(driver):
    rng = random.Random(2)
    nodes = [_entity(f'n{i}', 'G', _unit(rng)) for i in range(500)]
    for n in nodes:
        await n.save(driver)

    hits = total = 0
    for _ in range(20):
        q = _unit(rng)
        driver.use_vector_index = True
        ann = [
            n.uuid
            for n in await node_similarity_search(
                driver, q, SearchFilters(), ['G'], limit=10, min_score=-1.0
            )
        ]
        driver.use_vector_index = False
        exact = [
            n.uuid
            for n in await node_similarity_search(
                driver, q, SearchFilters(), ['G'], limit=10, min_score=-1.0
            )
        ]
        assert len(exact) == 10
        hits += len(set(ann) & set(exact))
        total += 10
    assert hits / total >= 0.9

    # ANN `score` and the exact in-query score are the same quantity, (1 + cosine) / 2
    q = _unit(rng)
    cosine = get_vector_cosine_func_query('n.name_embedding', '$q', driver.provider)
    ann_scores, _, _ = await driver.execute_query(
        "CALL db.index.vector.queryNodes('Entity[name_embedding]', 10, $q) "
        'YIELD node AS n, score RETURN n.uuid AS uuid, score',
        q=q,
    )
    exact_scores, _, _ = await driver.execute_query(
        f'MATCH (n:Entity) RETURN n.uuid AS uuid, {cosine} AS score', q=q
    )
    exact_by_uuid = {r['uuid']: r['score'] for r in exact_scores}
    assert ann_scores
    for r in ann_scores:
        assert r['score'] == pytest.approx(exact_by_uuid[r['uuid']], abs=1e-3)


async def test_similarity_with_label_and_group_filters(driver):
    """Label filter + group filter together once produced two WHERE clauses."""
    rng = random.Random(3)
    query = _axis(0)
    findings, plain = [], []
    for i in range(20):
        f = _entity(f'f{i}', 'G', _unit(rng, query, 0.1), ['Entity', 'Finding'])
        p = _entity(f'p{i}', 'G', _unit(rng, query, 0.1), ['Entity'])
        findings.append(f)
        plain.append(p)
        await f.save(driver)
        await p.save(driver)

    filters = SearchFilters(node_labels=['Finding'])
    uuids = await _both_paths_agree(
        node_similarity_search(driver, query, filters, ['G'], limit=5, min_score=0.0),
        driver.search_ops.node_similarity_search(
            driver, query, filters, ['G'], limit=5, min_score=0.0
        ),
    )
    assert len(uuids) == 5
    assert set(uuids) <= {n.uuid for n in findings}

    # asking for every finding: the candidate window (20 x 4) covers the whole tenant, so the
    # ANN result is already complete (the saturated-window re-scan is tested below)
    uuids = [
        n.uuid
        for n in await node_similarity_search(
            driver, query, filters, ['G'], limit=20, min_score=0.0
        )
    ]
    assert set(uuids) == {n.uuid for n in findings}


async def test_filtered_similarity_scans_exactly_only_when_the_window_hid_matches(driver):
    """Post-filters can starve the ANN candidate window. The exact scan is only worth its cost
    when the window was full and its weakest candidate still beat min_score; a small tenant or
    a min_score cut already gave the complete answer."""
    rng = random.Random(8)
    query = _axis(0)
    # 30 plain nodes crowd the 12-candidate window (limit 3 x overfetch 4); the 5 findings
    # are further away, outside it.
    for i in range(30):
        await _entity(f'p{i}', 'G', _unit(rng, query, 0.02)).save(driver)
    findings = [_entity(f'f{i}', 'G', _unit(rng), ['Entity', 'Finding']) for i in range(5)]
    for node in findings:
        await node.save(driver)
    filters = SearchFilters(node_labels=['Finding'])
    queries = _record_queries(driver)

    def exact_scans() -> int:
        return sum('vector.similarity.cosine' in q for q in queries)

    # window full of plain nodes, min_score 0: the findings are only reachable by the scan
    found = await node_similarity_search(driver, query, filters, ['G'], limit=3, min_score=0.0)
    assert len(found) == 3
    assert {n.uuid for n in found} <= {n.uuid for n in findings}
    assert exact_scans() == 1

    # same window, but min_score already cuts every candidate: nothing beyond can beat it
    queries.clear()
    found = await node_similarity_search(driver, query, filters, ['G'], limit=3, min_score=0.999)
    assert found == []
    assert exact_scans() == 0

    # the window covers the whole tenant: nothing was left out
    queries.clear()
    found = await node_similarity_search(driver, query, filters, ['G'], limit=20, min_score=0.0)
    assert {n.uuid for n in found} == {n.uuid for n in findings}
    assert exact_scans() == 0


async def test_edge_fulltext_and_similarity_after_a_real_save(driver):
    rng = random.Random(4)
    alice = _entity('Alice', 'G', _unit(rng))
    bob = _entity('Bob', 'G', _unit(rng))
    await alice.save(driver)
    await bob.save(driver)
    fact_embedding = _unit(rng)
    edge = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,
        group_id='G',
        name='LIKES',
        fact='Alice likes sourdough pizza',
        fact_embedding=fact_embedding,
        episodes=[],
        created_at=datetime.now(timezone.utc),
    )
    await edge.save(driver)

    live = await edge_fulltext_search(driver, 'sourdough', SearchFilters(), ['G'], limit=10)
    ops = await driver.search_ops.edge_fulltext_search(
        driver, 'sourdough', SearchFilters(), ['G'], limit=10
    )
    assert [e.uuid for e in live] == [e.uuid for e in ops] == [edge.uuid]

    other_tenant = await edge_fulltext_search(driver, 'sourdough', SearchFilters(), ['H'], limit=10)
    assert other_tenant == []

    live = await edge_similarity_search(
        driver, fact_embedding, None, None, SearchFilters(), ['G'], limit=10, min_score=0.5
    )
    ops = await driver.search_ops.edge_similarity_search(
        driver, fact_embedding, None, None, SearchFilters(), ['G'], limit=10, min_score=0.5
    )
    assert [e.uuid for e in live] == [e.uuid for e in ops] == [edge.uuid]


async def test_node_fulltext_is_tenant_scoped_and_matches_ops(driver):
    rng = random.Random(5)
    for group in ('A', 'B'):
        for i in range(5):
            await _entity(f'quokka {group}{i}', group, _unit(rng)).save(driver)

    live = await node_fulltext_search(driver, 'quokka', SearchFilters(), ['A'], limit=10)
    ops = await driver.search_ops.node_fulltext_search(
        driver, 'quokka', SearchFilters(), ['A'], limit=10
    )
    assert len(live) == 5
    assert {n.group_id for n in live} == {'A'}
    assert sorted(n.uuid for n in live) == sorted(n.uuid for n in ops)


async def test_community_similarity_matches_ops(driver):
    rng = random.Random(6)
    query = _axis(0)
    for g in ('A', 'B'):
        for i in range(10):
            emb = _unit(rng, query if g == 'B' else _axis(1), 0.05)
            await driver.execute_query(
                'CREATE (c:Community {uuid: $uuid, name: $name, group_id: $g, summary: "s", '
                'created_at: $now, name_embedding: $emb})',
                uuid=str(uuid4()),
                name=f'{g}{i}',
                g=g,
                now=datetime.now(timezone.utc),
                emb=emb,
            )

    live = await community_similarity_search(driver, query, ['A'], limit=5, min_score=0.0)
    ops = await driver.search_ops.community_similarity_search(
        driver, query, ['A'], limit=5, min_score=0.0
    )
    assert len(live) == 5
    assert {c.group_id for c in live} == {'A'}
    assert [c.uuid for c in live] == [c.uuid for c in ops]


async def test_clone_queries_the_other_tenants_database(driver):
    other_db = f'gi_it_{uuid4().hex[:10]}'
    _admin(f'create database {other_db}')
    try:
        rng = random.Random(7)
        mine = _entity('mine', 'G', _unit(rng))
        theirs = _entity('theirs', 'G', _unit(rng))
        clone = driver.clone(other_db)
        await mine.save(driver)
        await theirs.save(clone)

        assert (await EntityNode.get_by_uuid(driver, mine.uuid)).name == 'mine'
        assert (await EntityNode.get_by_uuid(clone, theirs.uuid)).name == 'theirs'
        records, _, _ = await clone.execute_query('MATCH (n:Entity) RETURN n.name AS name')
        assert [r['name'] for r in records] == ['theirs']
        records, _, _ = await driver.execute_query('MATCH (n:Entity) RETURN n.name AS name')
        assert [r['name'] for r in records] == ['mine']
    finally:
        _admin(f'drop database {other_db}')


async def test_bulk_save_round_trips_labels(driver, database):
    rng = random.Random(8)
    node = _entity('Finder', 'G', _unit(rng), ['Entity', 'Finding'])
    await add_nodes_and_edges_bulk(driver, [], [], [node], [], Mock(spec=EmbedderClient))

    loaded = await EntityNode.get_by_uuid(driver, node.uuid)
    assert set(loaded.labels) == {'Entity', 'Finding'}

    type_names = [t['name'] for t in _sql(database, 'SELECT name FROM schema:types')]
    assert not [n for n in type_names if '$(' in n], type_names


async def test_datetime_microseconds_survive_an_index_built_over_existing_data(database):
    """Engine #7164: a datetime index truncated to milliseconds. The index here is built
    AFTER the row exists, which is the case that used to lose precision."""
    d = _new_driver(database)
    try:
        created_at = datetime(2024, 3, 1, 12, 30, 45, 123456, tzinfo=timezone.utc)
        node = _entity('Dated', 'G', _unit(random.Random(9)), created_at=created_at)
        await node.save(d)

        await d.build_indices_and_constraints()

        loaded = await EntityNode.get_by_uuid(d, node.uuid)
        assert ensure_utc(loaded.created_at) == created_at
        records, _, _ = await d.execute_query(
            'MATCH (n:Entity) WHERE n.created_at = $at RETURN n.uuid AS uuid', at=created_at
        )
        assert [r['uuid'] for r in records] == [node.uuid]
    finally:
        await d.close()


async def test_build_indices_is_idempotent(driver, database, caplog):
    def index_names() -> list[str]:
        return sorted(r['name'] for r in _sql(database, 'SELECT name FROM schema:indexes'))

    before = index_names()
    assert any(n == 'node_name_and_summary' for n in before)

    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.arcadedb_driver'):
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints(delete_existing=False)

    assert index_names() == before
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_graphiti_group_id_provisions_a_missing_tenant_database(driver):
    """Graphiti turns every non-default group_id into driver.clone(group_id). That database
    does not exist yet and nothing else creates it, so the clone must, with its indexes,
    or the first add_episode(group_id=...) fails."""
    tenant = f'fresh_{uuid4().hex[:8]}'
    rng = random.Random(10)
    first = _entity('Zebra crossing', tenant, _unit(rng))
    second = _entity('Zebra herd', tenant, _unit(rng))
    embedder = Mock(spec=EmbedderClient)
    embedder.create.return_value = first.name_embedding
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
    )

    def exists() -> bool:
        response = httpx.get(f'{ARCADEDB_HTTP_URI}/api/v1/exists/{tenant}', auth=AUTH)
        return response.json()['result']

    try:
        assert not exists()
        group_id, scoped, _ = graphiti._resolve_request_scope(tenant)
        assert group_id == tenant

        await first.save(scoped)  # first use creates the database
        assert exists()

        # session path (bulk writes) works on the same, already provisioned, database
        await add_nodes_and_edges_bulk(scoped, [], [], [second], [], Mock(spec=EmbedderClient))

        index_names = {r['name'] for r in _sql(tenant, 'SELECT name FROM schema:indexes')}
        assert {'node_name_and_summary', 'edge_name_and_fact'} <= index_names

        found = await node_fulltext_search(scoped, 'zebra', SearchFilters(), [tenant], limit=10)
        assert {n.uuid for n in found} == {first.uuid, second.uuid}

        # Reads through Graphiti must reach the same per-group database the writes did,
        # not the driver's default one (which holds none of these nodes).
        config = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker=NodeReranker.rrf,
                sim_min_score=0.0,
            )
        )
        results = await graphiti.search_('zebra', config=config, group_ids=[tenant])
        assert {n.uuid for n in results.nodes} == {first.uuid, second.uuid}
        queries = _record_queries(scoped)
        near = await node_similarity_search(
            scoped, first.name_embedding, SearchFilters(), [tenant], limit=2, min_score=0.0
        )
        assert any('db.index.vector.queryNodes' in q for q in queries)
        assert near[0].uuid == first.uuid
    finally:
        if exists():
            _admin(f'drop database {tenant}')


async def test_similarity_search_before_the_indexes_exist_falls_back_to_the_exact_scan(database):
    """No build_indices_and_constraints(): first the Entity type does not exist, then it does
    but has no vector index. ArcadeDB words those two cases differently; both must fall back
    to the exact scan instead of raising."""
    d = _new_driver(database)
    try:
        query = _axis(0)
        assert await node_similarity_search(d, query, SearchFilters(), ['G'], 5, 0.0) == []

        node = _entity('Early', 'G', query)
        await node.save(d)  # creates the type, still no index
        found = await node_similarity_search(d, query, SearchFilters(), ['G'], 5, 0.0)
        assert [n.uuid for n in found] == [node.uuid]
    finally:
        await d.close()


async def test_reading_an_unknown_group_id_returns_nothing_and_creates_no_database(driver):
    """A search with a mistyped group_id used to create (and index) a database just to find
    nothing in it. Only a write creates a tenant database."""
    tenant = f'typo_{uuid4().hex[:8]}'
    scoped = driver.clone(tenant)

    def exists() -> bool:
        response = httpx.get(f'{ARCADEDB_HTTP_URI}/api/v1/exists/{tenant}', auth=AUTH)
        return response.json()['result']

    try:
        assert not exists()
        assert await node_fulltext_search(scoped, 'zebra', SearchFilters(), [tenant], limit=5) == []
        near = await node_similarity_search(
            scoped, _axis(0), SearchFilters(), [tenant], limit=5, min_score=0.0
        )
        assert near == []
        records, _, _ = await scoped.execute_query('MATCH (n) RETURN n', routing_='r')
        assert records == []
        assert not exists()

        with pytest.raises(ValueError, match='Invalid ArcadeDB database name'):
            await driver.clone('not a valid name!').execute_query('RETURN 1', routing_='r')

        node = _entity('Zebra', tenant, _axis(0))
        await node.save(scoped)  # the first write creates it, with its indexes
        assert exists()
        found = await node_fulltext_search(scoped, 'zebra', SearchFilters(), [tenant], limit=5)
        assert [n.uuid for n in found] == [node.uuid]
    finally:
        if exists():
            _admin(f'drop database {tenant}')


async def test_graphiti_read_apis_do_not_create_a_database_for_an_unknown_group_id(driver):
    """Every read has to carry the read hint, or the clone provisions a database for it."""
    from graphiti_core.edges import EntityEdge as Edge
    from graphiti_core.errors import GroupsEdgesNotFoundError
    from graphiti_core.nodes import EpisodicNode
    from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
    )
    reads = {
        'retrieve_episodes': lambda s, t: retrieve_episodes(
            s, datetime.now(timezone.utc), group_ids=[t]
        ),
        'Graphiti.retrieve_episodes': lambda s, t: graphiti.retrieve_episodes(
            datetime.now(timezone.utc), group_ids=[t]
        ),
        'EntityNode.get_by_group_ids': lambda s, t: EntityNode.get_by_group_ids(s, [t]),
        'EpisodicNode.get_by_group_ids': lambda s, t: EpisodicNode.get_by_group_ids(s, [t]),
        'EntityEdge.get_by_group_ids': lambda s, t: Edge.get_by_group_ids(s, [t]),
    }
    created = []
    for name, read in reads.items():
        tenant = f'typo_{uuid4().hex[:8]}'
        # the edge reader reports an empty result by raising
        with contextlib.suppress(GroupsEdgesNotFoundError):
            assert await read(driver.clone(tenant), tenant) == []
        if httpx.get(f'{ARCADEDB_HTTP_URI}/api/v1/exists/{tenant}', auth=AUTH).json()['result']:
            created.append(name)
            _admin(f'drop database {tenant}')
    assert created == []


async def test_fulltext_with_several_group_ids_returns_every_listed_tenant(driver):
    """The group clause is `group_id:"A" OR group_id:"B" AND (q)`; tenant scoping must not
    depend on Lucene's operator precedence."""
    rng = random.Random(11)
    for group in ('tenant_a', 'tenant-b', 'tenant_c'):
        for i in range(3):
            await _entity(f'okapi {group} {i}', group, _unit(rng)).save(driver)

    found = await node_fulltext_search(
        driver, 'okapi', SearchFilters(), ['tenant_a', 'tenant-b'], limit=20
    )
    assert {n.group_id for n in found} == {'tenant_a', 'tenant-b'}
    assert len(found) == 6


async def test_non_utc_datetimes_are_stored_as_the_same_instant(driver):
    plus_two = timezone(timedelta(hours=2))
    created_at = datetime(2024, 3, 1, 14, 0, 0, 5, tzinfo=plus_two)
    node = _entity('Zoned', 'G', _unit(random.Random(12)), created_at=created_at)
    await node.save(driver)

    loaded = await EntityNode.get_by_uuid(driver, node.uuid)
    assert ensure_utc(loaded.created_at) == created_at.astimezone(timezone.utc)


EDGE_TYPES = [
    ('RELATES_TO', 'Entity', 'Entity'),
    ('MENTIONS', 'Episodic', 'Entity'),
    ('HAS_MEMBER', 'Community', ''),
    ('HAS_EPISODE', 'Saga', 'Episodic'),
    ('NEXT_EPISODE', 'Episodic', 'Episodic'),
]


async def _plan(driver, query: str, **params) -> str:
    result = await driver.client.execute_query(
        'PROFILE ' + query, parameters_=params, database_=driver._database
    )
    return (result.summary.profile or {}).get('args', {}).get('string-representation', '')


@pytest.mark.parametrize(('rel_type', 'source_label', 'target_label'), EDGE_TYPES)
async def test_edge_uuid_lookups_seek_the_relationship_index(
    driver, rel_type, source_label, target_label
):
    """Graphiti looks edges up by uuid with an inline ``{uuid: $uuid}`` map, labelled endpoints and
    a ``T1|T2`` union for deletes, and saves them with a nested ``$edge_data.field`` parameter.
    Before ArcadeData/arcadedb#11480 and #11481 (fixed by #11486) ArcadeDB planned each of those
    as a scan of every edge or vertex (~100 ms at 20k edges). This test needs a server with that
    fix, and fails on one without it (the rows are still right, only slow)."""
    # endpoints carry the labels the pattern asks for, so the labelled shapes match real nodes
    await driver.execute_query(
        f'CREATE (:{source_label or "Entity"} {{uuid: "s"}}), (:{target_label or "Entity"} {{uuid: "t"}})'
    )
    for uuid in ('e1', 'e2', 'e3'):
        await driver.execute_query(
            f'MATCH (a {{uuid: "s"}}), (b {{uuid: "t"}}) CREATE (a)-[:{rel_type} {{uuid: $uuid}}]->(b)',
            uuid=uuid,
        )
    source = f'n:{source_label}' if source_label else 'n'
    target = f'm:{target_label}' if target_label else 'm'
    shapes = {
        'inline map': (
            f'MATCH ({source})-[e:{rel_type} {{uuid: $uuid}}]->({target}) RETURN e.uuid',
            {'uuid': 'e1'},
        ),
        'IN list': (
            f'MATCH ({source})-[e:{rel_type}]->({target}) WHERE e.uuid IN $uuids RETURN e.uuid',
            {'uuids': ['e1', 'e2']},
        ),
        'type union': (
            f'MATCH (n)-[e:{rel_type}|MENTIONS {{uuid: $uuid}}]->(m) RETURN e.uuid',
            {'uuid': 'e1'},
        ),
    }

    plans = {name: await _plan(driver, query, **params) for name, (query, params) in shapes.items()}
    for name, plan in plans.items():
        assert 'MATCH EDGE BY INDEX' in plan, (name, plan)
        assert 'NodeByLabelScan' not in plan, (name, plan)


async def test_edge_save_endpoints_are_looked_up_through_the_uuid_index(driver):
    """The save statement passes the endpoints as ``$edge_data.source_uuid``; ArcadeDB used to plan
    that as a label scan (ArcadeData/arcadedb#11480, fixed by #11486); this fails on a server without it."""
    from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_query

    rng = random.Random(13)
    source, target = _entity('S', 'G', _unit(rng)), _entity('T', 'G', _unit(rng))
    await source.save(driver)
    await target.save(driver)
    edge_data = {
        'uuid': str(uuid4()),
        'source_uuid': source.uuid,
        'target_uuid': target.uuid,
        'name': 'KNOWS',
        'fact': 'S knows T',
        'group_id': 'G',
    }

    result = await driver.client.execute_query(
        'PROFILE ' + get_entity_edge_save_query(driver.provider),
        parameters_={'edge_data': edge_data},
        database_=driver._database,
    )
    plan = (result.summary.profile or {}).get('args', {}).get('string-representation', '')
    assert 'NodeByLabelScan' not in plan, plan


async def test_edge_uuid_lookups_return_and_delete_every_edge_type(driver):
    from datetime import datetime, timezone

    from graphiti_core.edges import (
        CommunityEdge,
        Edge,
        EpisodicEdge,
        HasEpisodeEdge,
        NextEpisodeEdge,
    )
    from graphiti_core.nodes import (
        CommunityNode,
        EpisodeType,
        EpisodicNode,
        SagaNode,
    )

    now = datetime.now(timezone.utc)
    rng = random.Random(5)
    source, target = _entity('S', 'G', _unit(rng)), _entity('T', 'G', _unit(rng))
    episode = EpisodicNode(
        name='ep',
        group_id='G',
        source=EpisodeType.text,
        source_description='d',
        content='c',
        valid_at=now,
    )
    next_episode = episode.model_copy(update={'uuid': str(uuid4()), 'name': 'ep2'})
    community = CommunityNode(name='C', group_id='G', summary='s', name_embedding=_unit(rng))
    saga = SagaNode(name='saga', group_id='G')
    for node in (source, target, episode, next_episode, community, saga):
        await node.save(driver)

    entity_edge = EntityEdge(
        group_id='G',
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='KNOWS',
        fact='S knows T',
        fact_embedding=_unit(rng),
        episodes=[],
        created_at=now,
    )
    others = [
        EpisodicEdge(
            group_id='G',
            source_node_uuid=episode.uuid,
            target_node_uuid=source.uuid,
            created_at=now,
        ),
        CommunityEdge(
            group_id='G',
            source_node_uuid=community.uuid,
            target_node_uuid=source.uuid,
            created_at=now,
        ),
        HasEpisodeEdge(
            group_id='G', source_node_uuid=saga.uuid, target_node_uuid=episode.uuid, created_at=now
        ),
        NextEpisodeEdge(
            group_id='G',
            source_node_uuid=episode.uuid,
            target_node_uuid=next_episode.uuid,
            created_at=now,
        ),
    ]
    edges = [entity_edge, *others]
    for edge in edges:
        await edge.save(driver)

    for edge in edges:
        loaded = await type(edge).get_by_uuid(driver, edge.uuid)
        assert (loaded.uuid, loaded.source_node_uuid, loaded.target_node_uuid) == (
            edge.uuid,
            edge.source_node_uuid,
            edge.target_node_uuid,
        )
        assert [e.uuid for e in await type(edge).get_by_uuids(driver, [edge.uuid])] == [edge.uuid]

    reloaded = await EntityEdge.get_by_uuid(driver, entity_edge.uuid)
    reloaded.fact_embedding = None
    await reloaded.load_fact_embedding(driver)
    assert reloaded.fact_embedding == pytest.approx(entity_edge.fact_embedding, abs=1e-6)

    ops = driver.entity_edge_ops
    assert (await ops.get_by_uuid(driver, entity_edge.uuid)).uuid == entity_edge.uuid
    assert [e.uuid for e in await ops.get_by_uuids(driver, [entity_edge.uuid])] == [
        entity_edge.uuid
    ]
    await ops.load_embeddings_bulk(driver, [reloaded])

    # Edge.delete / delete_by_uuids are type-agnostic only for these three edge types; the saga
    # and next-episode edges are deleted by their own classes.
    await ops.delete(driver, edges[0])
    await Edge.delete_by_uuids(driver, [edges[1].uuid])
    await driver.community_edge_ops.delete_by_uuids(driver, [edges[2].uuid])
    await edges[3].delete(driver)
    await driver.next_episode_edge_ops.delete(driver, edges[4])
    remaining, _, _ = await driver.execute_query(
        'MATCH ()-[e]->() RETURN count(e) AS n', routing_='r'
    )
    assert remaining[0]['n'] == 0


@pytest.mark.parametrize('path', ['EntityEdge.save', 'add_nodes_and_edges_bulk'])
async def test_concurrent_saves_of_the_same_edge_store_it_once(driver, path):
    """Several workers saving the same edge at once must store it once. ArcadeDB's MERGE was not
    atomic against a concurrent MERGE of the same relationship (ArcadeData/arcadedb#11482, fixed
    by #11486), which the driver's write gate used to hide; this is the regression test."""
    from datetime import datetime, timezone

    embedder = Mock(spec=EmbedderClient)
    rng = random.Random(21)
    source, target = _entity('S', 'G', _unit(rng)), _entity('T', 'G', _unit(rng))
    await add_nodes_and_edges_bulk(driver, [], [], [source, target], [], embedder)
    edges = [
        EntityEdge(
            group_id='G',
            source_node_uuid=source.uuid,
            target_node_uuid=target.uuid,
            name='KNOWS',
            fact=f'fact {i}',
            fact_embedding=_unit(rng),
            episodes=[],
            created_at=datetime.now(timezone.utc),
        )
        for i in range(60)
    ]

    async def save(edge: EntityEdge) -> None:
        if path == 'EntityEdge.save':
            await edge.save(driver)
        else:
            await add_nodes_and_edges_bulk(driver, [], [], [], [edge], embedder)

    # four workers ingest each fact at the same moment
    await asyncio.gather(*[save(edge) for edge in edges for _ in range(4)])

    records, _, _ = await driver.execute_query(
        'MATCH ()-[e:RELATES_TO]->() RETURN e.uuid AS uuid', routing_='r'
    )
    assert sorted(r['uuid'] for r in records) == sorted(e.uuid for e in edges)
