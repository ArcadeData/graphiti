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

import logging
from typing import Any, cast

from typing_extensions import LiteralString

from graphiti_core.driver.query_executor import QueryExecutor

logger = logging.getLogger(__name__)

# When a similarity search carries predicates other than group_id, they can only be
# applied after the ANN call, so ask the index for this many times `limit` candidates to
# leave room for them to discard some.
ARCADEDB_ANN_OVERFETCH = 4


def vector_index_name(label: str, prop: str = 'name_embedding') -> str:
    """Name under which ``db.index.vector.queryNodes`` addresses an index."""
    return f'{label}[{prop}]'


def ann_query(
    *,
    label: str,
    alias: str,
    return_query: str,
    filter_queries: list[str],
    by_group: bool,
    limit: int,
) -> tuple[LiteralString, int]:
    """Build the ANN similarity query and the ``k`` to request from the index.

    ``group_id`` is pushed into the index call (``where``) so it is applied *during* the
    search: a plain top-k followed by a group filter returns fewer than ``limit`` rows,
    or none, whenever other tenants' vectors are closer. Any other predicate in
    ``filter_queries`` runs after the call, hence the over-fetch.

    ``score`` from ``queryNodes`` is already ``(1 + cosine) / 2``, the scale of
    ``get_vector_cosine_func_query`` and of ``min_score``.

    Params the query expects: ``search_vector``, ``k``, ``limit``, ``min_score``, plus
    ``group_ids`` when ``by_group`` and whatever ``filter_queries`` reference.
    """
    k = limit * ARCADEDB_ANN_OVERFETCH if filter_queries else limit
    where = ' AND '.join(['score > $min_score', *filter_queries])
    query = (
        f'{_ann_call(label, by_group)} '
        f'YIELD node AS {alias}, score '
        f'WITH {alias}, score WHERE {where} '
        f'RETURN {return_query} '
        'ORDER BY score DESC LIMIT $limit'
    )
    return cast(LiteralString, query), k


def _vector_index_unavailable(error: Exception, label: str) -> bool:
    """Whether ``error`` says the vector index (or its type) does not exist yet.

    ArcadeDB 26.11.1 words it "No vector index found on property 'name_embedding' for type
    'Entity'" when the type exists and "Type with name 'Entity' was not found" when it does
    not, as happens when ``build_indices_and_constraints()`` has not run.
    """
    message = str(error)
    return (
        vector_index_name(label) in message
        or 'No vector index found' in message
        or f"Type with name '{label}' was not found" in message
    )


def _ann_call(label: str, by_group: bool) -> str:
    options = ", {where: 'group_id IN :group_ids'}" if by_group else ''
    return (
        f"CALL db.index.vector.queryNodes('{vector_index_name(label)}', $k, "
        f'$search_vector{options})'
    )


def ann_window_query(*, label: str, by_group: bool) -> LiteralString:
    """How many candidates the index returns for ``k`` and how weak the last one is.

    Params: ``search_vector``, ``k`` and ``group_ids`` when ``by_group``.
    """
    query = (
        f'{_ann_call(label, by_group)} YIELD score '
        'RETURN count(score) AS candidates, min(score) AS lowest'
    )
    return cast(LiteralString, query)


async def ann_search(
    executor: QueryExecutor,
    *,
    label: str,
    alias: str,
    return_query: str,
    search_vector: list[float],
    filter_queries: list[str],
    filter_params: dict[str, Any],
    group_ids: list[str] | None,
    limit: int,
    min_score: float,
) -> list[Any] | None:
    """Run the ANN query; ``None`` means "use the exact in-query scan instead".

    That is the case when the index does not exist (``build_indices_and_constraints``
    was not run, or ``use_vector_index`` was off at creation) or when post-filters left
    fewer than ``limit`` rows of the over-fetched candidates, where an exact scan can
    still find matches the candidate window missed.
    """
    query, k = ann_query(
        label=label,
        alias=alias,
        return_query=return_query,
        filter_queries=filter_queries,
        by_group=group_ids is not None,
        limit=limit,
    )
    params: dict[str, Any] = dict(filter_params)
    if group_ids is not None:
        params['group_ids'] = group_ids

    try:
        records, _, _ = await executor.execute_query(
            query,
            search_vector=search_vector,
            k=k,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **params,
        )
    except Exception as e:
        if _vector_index_unavailable(e, label):
            logger.warning(
                f'ArcadeDB vector index {vector_index_name(label)} unavailable, falling back '
                f'to exact similarity search: {e}'
            )
            return None
        raise

    if filter_queries and len(records) < limit:
        # Short result. The exact scan can only find more if candidates were left outside the
        # window: it was full (k candidates) and its weakest one still beats min_score, since
        # everything beyond it scores lower. Otherwise the index was exhausted (a small tenant)
        # or min_score ended the list, and the answer is already complete.
        window, _, _ = await executor.execute_query(
            ann_window_query(label=label, by_group=group_ids is not None),
            search_vector=search_vector,
            k=k,
            routing_='r',
            **({'group_ids': group_ids} if group_ids is not None else {}),
        )
        candidates, lowest = window[0]['candidates'], window[0]['lowest']
        if candidates >= k and lowest > min_score:
            return None

    return list(records)
