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

Live tests against an ArcadeDB server (26.10.1+) with the Bolt plugin enabled.
Each test gets its own throwaway database, created and dropped over the HTTP API at
ARCADEDB_HTTP_URI (see tests/helpers_test.py).
"""

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

    # the fallback path: more is asked of the post-filter than the candidate window holds
    uuids = [
        n.uuid
        for n in await node_similarity_search(
            driver, query, filters, ['G'], limit=20, min_score=0.0
        )
    ]
    assert set(uuids) == {n.uuid for n in findings}


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


async def test_edge_save_endpoints_are_looked_up_through_the_uuid_index(driver):
    """The save statement must seek the Entity uuid index; a nested parameter field turns
    each endpoint MATCH into a scan of every Entity."""
    from graphiti_core.models.edges.edge_db_queries import (
        get_entity_edge_save_params,
        get_entity_edge_save_query,
    )

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
        parameters_=get_entity_edge_save_params(driver.provider, edge_data),
        database_=driver._database,
    )
    plan = (result.summary.profile or {}).get('args', {}).get('string-representation', '')

    assert 'NodeIndexSeek' in plan, plan
    assert 'NodeByLabelScan' not in plan, plan
