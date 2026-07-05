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

from typing import Any

from graphiti_core.driver.arcadedb.operations.search_ops import ArcadeDBSearchOperations
from graphiti_core.driver.search_interface.search_interface import SearchInterface


class ArcadeDBSearchInterface(SearchInterface):
    """Routes the fulltext/similarity search dispatchers in
    graphiti_core.search.search_utils to ArcadeDBSearchOperations instead of
    the generic Neo4j-fulltext-procedure path.

    ArcadeDB does not support the `db.index.fulltext.query{Nodes,Relationships}`
    Bolt procedures that search_utils.py's generic (non-Neptune/Kuzu) branch
    assumes, so edge/node/episode/community fulltext search crash for ArcadeDB
    today. ArcadeDBSearchOperations already implements the CONTAINS-based
    equivalent correctly -- it was just never wired up. `GraphDriver` extends
    `QueryExecutor`, so the driver instance can be passed directly as the
    `executor` argument these methods expect.

    6 of the 12 methods SearchInterface declares are implemented here:
    edge_fulltext_search, edge_similarity_search, node_fulltext_search,
    node_similarity_search and episode_fulltext_search are implemented because
    search_utils.py calls them WITHOUT a `try/except NotImplementedError` guard
    around the `search_interface` dispatch -- leaving them unimplemented would
    turn "search_interface not set" (falls through to the generic path) into
    "search_interface set but incomplete" (crashes with an uncaught
    NotImplementedError) for functionality that already works today via the
    generic path. community_fulltext_search IS guarded, but is additionally
    implemented anyway because its generic fallback has the identical
    fulltext-procedure bug. The remaining 6 methods (edge_bfs_search,
    node_bfs_search, community_similarity_search, node_distance_reranker,
    episode_mentions_reranker, get_embeddings_for_communities) are
    intentionally left unimplemented: their generic paths use provider-agnostic
    Cypher (no fulltext procedure) and already work correctly against
    ArcadeDB, and all of their call sites in search_utils.py DO guard with
    `except NotImplementedError: pass`, so they safely keep using that
    already-working path.
    """

    def __init__(self):
        self._ops = ArcadeDBSearchOperations()

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await self._ops.edge_fulltext_search(driver, query, search_filter, group_ids, limit)

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        return await self._ops.edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids,
            limit,
            min_score,
        )

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await self._ops.node_fulltext_search(driver, query, search_filter, group_ids, limit)

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        return await self._ops.node_similarity_search(
            driver, search_vector, search_filter, group_ids, limit, min_score
        )

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await self._ops.episode_fulltext_search(
            driver, query, search_filter, group_ids, limit
        )

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await self._ops.community_fulltext_search(driver, query, group_ids, limit)
