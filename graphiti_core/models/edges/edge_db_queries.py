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

from graphiti_core.driver.driver import GraphProvider

EPISODIC_EDGE_SAVE = """
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MATCH (node:Entity {uuid: $entity_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""


def get_episodic_edge_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (episode:Episodic {uuid: $source_node_uuid})
            MATCH (node:Entity {uuid: $target_node_uuid})
            MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at
            RETURN e.uuid AS uuid
        """

    return """
        UNWIND $episodic_edges AS edge
        MATCH (episode:Episodic {uuid: edge.source_node_uuid})
        MATCH (node:Entity {uuid: edge.target_node_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
        SET
            e.group_id = edge.group_id,
            e.created_at = edge.created_at
        RETURN e.uuid AS uuid
    """


EPISODIC_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_entity_edge_save_params(
    provider: GraphProvider, edge_data: dict[str, Any]
) -> dict[str, Any]:
    """Query parameters for get_entity_edge_save_query."""
    params: dict[str, Any] = {'edge_data': edge_data}
    if provider == GraphProvider.ARCADEDB:
        params['source_uuid'] = edge_data['source_uuid']
        params['target_uuid'] = edge_data['target_uuid']
        params['edge_uuid'] = edge_data['uuid']
    return params


def get_entity_edge_save_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.ARCADEDB:
            # ArcadeDB: embeddings stored as regular properties, no vector API call.
            # The endpoints and the edge uuid are flat parameters (see
            # get_entity_edge_save_params): ArcadeDB plans
            # `MATCH (n:Entity {uuid: $edge_data.source_uuid})` as a label scan with a filter
            # (~35 ms per MATCH at 20k nodes, growing with the node count), while a flat
            # parameter is a NodeIndexSeek (~1 ms).
            return """
                MATCH (source:Entity {uuid: $source_uuid})
                MATCH (target:Entity {uuid: $target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_uuid}]->(target)
                SET e = $edge_data
                RETURN e.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = $edge_data
                SET e.fact_embedding = vecf32($edge_data.fact_embedding)
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (source:Entity {uuid: $edge_data.source_uuid})
                MATCH (target:Entity {uuid: $edge_data.target_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                SET e = removeKeyFromMap(removeKeyFromMap($edge_data, "fact_embedding"), "episodes")
                SET e.fact_embedding = join([x IN coalesce($edge_data.fact_embedding, []) | toString(x) ], ",")
                SET e.episodes = join($edge_data.episodes, ",")
                RETURN $edge_data.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (source:Entity {uuid: $source_uuid})
                MATCH (target:Entity {uuid: $target_uuid})
                MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at,
                    e.name = $name,
                    e.fact = $fact,
                    e.fact_embedding = $fact_embedding,
                    e.episodes = $episodes,
                    e.expired_at = $expired_at,
                    e.valid_at = $valid_at,
                    e.invalid_at = $invalid_at,
                    e.reference_time = $reference_time,
                    e.attributes = $attributes
                RETURN e.uuid AS uuid
            """
        case _:  # Neo4j
            save_embedding_query = (
                """WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)"""
                if not has_aoss
                else ''
            )
            return (
                (
                    """
                        MATCH (source:Entity {uuid: $edge_data.source_uuid})
                        MATCH (target:Entity {uuid: $edge_data.target_uuid})
                        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
                        SET e = $edge_data
                        """
                    + save_embedding_query
                )
                + """
                RETURN e.uuid AS uuid
                """
            )


def get_entity_edge_save_bulk_query(provider: GraphProvider, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.ARCADEDB:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
                SET e = edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = edge
                SET r.fact_embedding = vecf32(edge.fact_embedding)
                WITH r, edge
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                UNWIND $entity_edges AS edge
                MATCH (source:Entity {uuid: edge.source_node_uuid})
                MATCH (target:Entity {uuid: edge.target_node_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
                SET r = removeKeyFromMap(removeKeyFromMap(edge, "fact_embedding"), "episodes")
                SET r.fact_embedding = join([x IN coalesce(edge.fact_embedding, []) | toString(x) ], ",")
                SET r.episodes = join(edge.episodes, ",")
                RETURN edge.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (source:Entity {uuid: $source_node_uuid})
                MATCH (target:Entity {uuid: $target_node_uuid})
                MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at,
                    e.name = $name,
                    e.fact = $fact,
                    e.fact_embedding = $fact_embedding,
                    e.episodes = $episodes,
                    e.expired_at = $expired_at,
                    e.valid_at = $valid_at,
                    e.invalid_at = $invalid_at,
                    e.reference_time = $reference_time,
                    e.attributes = $attributes
                RETURN e.uuid AS uuid
            """
        case _:
            save_embedding_query = (
                'WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)'
                if not has_aoss
                else ''
            )
            return (
                """
                    UNWIND $entity_edges AS edge
                    MATCH (source:Entity {uuid: edge.source_node_uuid})
                    MATCH (target:Entity {uuid: edge.target_node_uuid})
                    MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
                    SET e = edge
                    """
                + save_embedding_query
                + """
                RETURN edge.uuid AS uuid
            """
            )


def get_entity_edge_return_query(provider: GraphProvider) -> str:
    # `fact_embedding` is not returned by default and must be manually loaded using `load_fact_embedding()`.
    # Direction semantics: NEO4J/FALKORDB/NEPTUNE read source/target from the relationship itself
    # via Cypher's built-in `startNode(e)`/`endNode(e)`, so the returned direction matches the
    # stored relationship regardless of whether the surrounding MATCH is directed or undirected.
    # KUZU models relationships as a `RelatesToNode_` intermediate node plus two `:RELATES_TO`
    # edges, so `e` is a node, not a relationship, and `startNode`/`endNode` are not applicable.
    # KUZU's MATCH is always directed `(n)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m)`,
    # so `n`/`m` already reflect the true source/target.

    if provider == GraphProvider.KUZU:
        return """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        e.attributes AS attributes
    """

    if provider == GraphProvider.NEPTUNE:
        return """
        e.uuid AS uuid,
        startNode(e).uuid AS source_node_uuid,
        endNode(e).uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.name AS name,
        e.fact AS fact,
        split(e.episodes, ',') AS episodes,
        e.created_at AS created_at,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        properties(e) AS attributes
    """

    return """
        e.uuid AS uuid,
        startNode(e).uuid AS source_node_uuid,
        endNode(e).uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        properties(e) AS attributes
    """


def get_community_edge_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.ARCADEDB:
            # ArcadeDB: use label-less MATCH for target node (matches any type)
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node {uuid: $entity_uuid})
                WHERE node:Entity OR node:Community
                MERGE (community)-[r:HAS_MEMBER {uuid: $uuid}]->(node)
                SET r.uuid= $uuid
                SET r.group_id= $group_id
                SET r.created_at= $created_at
                RETURN r.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Entity {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at
                RETURN e.uuid AS uuid
                UNION
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Community {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET
                    e.group_id = $group_id,
                    e.created_at = $created_at
                RETURN e.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MATCH (community:Community {uuid: $community_uuid})
                MATCH (node:Entity | Community {uuid: $entity_uuid})
                MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
                SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
                RETURN e.uuid AS uuid
            """


COMMUNITY_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


HAS_EPISODE_EDGE_SAVE = """
    MATCH (saga:Saga {uuid: $saga_uuid})
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MERGE (saga)-[e:HAS_EPISODE {uuid: $uuid}]->(episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

HAS_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


NEXT_EPISODE_EDGE_SAVE = """
    MATCH (source_episode:Episodic {uuid: $source_episode_uuid})
    MATCH (target_episode:Episodic {uuid: $target_episode_uuid})
    MERGE (source_episode)-[e:NEXT_EPISODE {uuid: $uuid}]->(target_episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

NEXT_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_edge_match(
    provider: GraphProvider, rel_type: str, source_label: str = '', target_label: str = ''
) -> str:
    """``MATCH (n)-[e:TYPE]->(m)`` binding ``n``, ``e`` and ``m``, with the endpoint labels.

    ArcadeDB gets anonymous endpoints: it only plans its relationship uuid index
    (``MATCH EDGE BY INDEX``, ~0.4 ms) for a single relationship type with unlabelled
    endpoints plus ``WHERE e.uuid = ...`` / ``IN``. A label on either endpoint, a
    ``T1|T2`` union or an inline ``{uuid: ...}`` map is planned as a scan of every node or
    edge (~100 ms at 20k edges). Each edge type connects one label pair, so the labels carry
    no information there.
    """
    if provider == GraphProvider.ARCADEDB:
        return f'MATCH (n)-[e:{rel_type}]->(m)'
    source = f'n:{source_label}' if source_label else 'n'
    target = f'm:{target_label}' if target_label else 'm'
    return f'MATCH ({source})-[e:{rel_type}]->({target})'


def get_edge_by_uuid_match(
    provider: GraphProvider, rel_type: str, source_label: str = '', target_label: str = ''
) -> str:
    """MATCH clause that binds ``n``, ``e`` and ``m`` to the edge whose uuid is ``$uuid``."""
    if provider == GraphProvider.ARCADEDB:
        return f'{get_edge_match(provider, rel_type)}\nWHERE e.uuid = $uuid'
    source = f'n:{source_label}' if source_label else 'n'
    target = f'm:{target_label}' if target_label else 'm'
    return f'MATCH ({source})-[e:{rel_type} {{uuid: $uuid}}]->({target})'


def get_edges_by_uuids_match(
    provider: GraphProvider, rel_type: str, source_label: str = '', target_label: str = ''
) -> str:
    """MATCH clause that binds ``n``, ``e`` and ``m`` to the edges whose uuid is in ``$uuids``."""
    match = get_edge_match(provider, rel_type, source_label, target_label)
    return f'{match}\nWHERE e.uuid IN $uuids'


# Edge types a uuid can belong to when the caller does not know which one it has (delete()).
ARCADEDB_UUID_DELETE_REL_TYPES = ('MENTIONS', 'RELATES_TO', 'HAS_MEMBER')


def get_arcadedb_edge_delete_queries(by_uuids: bool = False) -> list[str]:
    """One DELETE per edge type.

    ArcadeDB only reaches its relationship uuid index with a single relationship type and
    anonymous endpoints (``MATCH (n)-[e:T]->(m) WHERE e.uuid = $uuid``); an inline property
    map, a labelled endpoint or a ``T1|T2|T3`` union each fall back to scanning every edge or
    every node (~100 ms at 20k edges, against ~0.4 ms for the index).
    """
    match = get_edges_by_uuids_match if by_uuids else get_edge_by_uuid_match
    return [
        f'{match(GraphProvider.ARCADEDB, rel_type)}\nDELETE e'
        for rel_type in ARCADEDB_UUID_DELETE_REL_TYPES
    ]
