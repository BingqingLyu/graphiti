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
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neug import dialect
from graphiti_core.driver.neug.dialect import neug_in_filter
from graphiti_core.driver.neug.parsers import (
    community_node_from_record,
    entity_edge_from_record,
    entity_node_from_record,
    episodic_node_from_record,
)
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_doc_return_query,
    get_entity_edge_return_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    EPISODIC_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    neug_node_label_filter,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)


class NeuGSearchOperations(SearchOperations):
    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        fuzzy_query = dialect.build_fts_query(query)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG
        )

        if group_ids is not None:
            filter_queries.append(neug_in_filter('n.group_id', group_ids, filter_params))

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Single weighted multi-property bm25 over the Entity(name, summary)
        # FTS index (name weighted 3x summary). group/label filters ride the
        # PRE-bm25 WHERE, so the top-K is computed within the filtered set
        # (exact recall). Replaces two per-column queries merged client-side,
        # which also ran the filter AFTER ORDER BY/LIMIT (a recall bug).
        records, _, _ = await executor.execute_query(
            """
            MATCH (n:Entity)
            """
            + filter_query
            + """
            WITH n, bm25([n.name, n.summary], [3.0, 1.0], $query) AS score
            ORDER BY score ASC LIMIT """
            + str(int(limit))
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEUG),
            query=fuzzy_query,
            **filter_params,
        )

        return [entity_node_from_record(r) for r in records]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG
        )

        if group_ids is not None:
            filter_queries.append(neug_in_filter('n.group_id', group_ids, filter_params))

        # HNSW ANN path: a terminal
        #   RETURN ..., vector_distance_cosine(n.name_embedding, <lit>) AS dist
        #   ORDER BY dist ASC LIMIT k
        # is rewritten into an ANN IndexScan. group/label filters ride the
        # pre-RETURN WHERE (pushed into the scan as a scalar pre-filter, so the
        # top-K is computed within the filtered set - exact recall). The old
        # extra `WITH n, dist` projection turned dist into a plain variable
        # reference the rewrite no longer recognizes, degrading to a
        # brute-force scan. min_score trims the ranked top-K client-side
        # (monotone in dist). The query vector rides a bound $search_vector:
        # inlining it makes every query's text unique, so NeuG re-plans each
        # call and that compilation is ~92% of the latency (see
        # search_utils.node_similarity_search for the measurement).
        pre_filter = (' WHERE ' + ' AND '.join(filter_queries)) if filter_queries else ''
        cypher = (
            """
            MATCH (n:Entity)
            """
            + pre_filter
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEUG)
            + """,
            vector_distance_cosine(n.name_embedding, $search_vector) AS dist
            ORDER BY dist ASC LIMIT """
            + str(int(limit))
        )

        records, _, _ = await executor.execute_query(
            cypher,
            search_vector=search_vector,
            **filter_params,
        )

        return [entity_node_from_record(r) for r in records if (1.0 - float(r['dist'])) > min_score]

    async def node_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        search_filter: SearchFilters,
        max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        if not origin_uuids or max_depth < 1:
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG
        )

        if group_ids is not None:
            filter_queries.append(neug_in_filter('n.group_id', group_ids, filter_params))
            filter_queries.append(neug_in_filter('origin.group_id', group_ids, filter_params))

        # Each NEUG leg already carries a WHERE clause, so extra filters join
        # on with AND rather than introducing a second WHERE.
        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        # No UNWIND and no mixed-type traversal from an unlabelled origin, so
        # BFS runs as one leg per origin kind; MENTIONS only ever appears as
        # the first hop out of an Episodic origin.
        origins = neug_in_filter('origin.uuid', origin_uuids, filter_params)
        match_queries = [
            f"""
            MATCH (origin:Entity)-[:RELATES_TO*1..{max_depth}]->(n:Entity)
            WHERE {origins}
            AND n.group_id = origin.group_id
            """,
        ]
        if max_depth == 1:
            match_queries.append(f"""
                MATCH (origin:Episodic)-[:MENTIONS]->(n:Entity)
                WHERE {origins}
                AND n.group_id = origin.group_id
            """)
        else:
            match_queries.append(f"""
                MATCH (origin:Episodic)-[:MENTIONS]->(:Entity)-[:RELATES_TO*0..{max_depth - 1}]->(n:Entity)
                WHERE {origins}
                AND n.group_id = origin.group_id
            """)

        # Dedupe engine-side: variable-length patterns enumerate every path,
        # which explodes combinatorially over bidirectional edges and hubs;
        # per-path rows also re-serialize the full RETURN payload each time.
        records: list[dict[str, Any]] = []
        seen_uuids: set[str] = set()
        for match_query in match_queries:
            sub_records, _, _ = await executor.execute_query(
                match_query
                + filter_query
                + """
                WITH DISTINCT n
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.NEUG)
                + """
                LIMIT """
                + str(int(limit)),
                **filter_params,
            )
            for record in sub_records:
                if record['uuid'] not in seen_uuids:
                    seen_uuids.add(record['uuid'])
                    records.append(record)

        return [entity_node_from_record(r) for r in records]

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        fuzzy_query = dialect.build_fts_query(query)
        if fuzzy_query == '':
            return []

        # Rank on the EdgeDoc mirror's FTS index with bm25() (negative scores,
        # ascending = more relevant) and project the edge straight from
        # EdgeDoc, which carries every returned field plus both endpoint uuids,
        # so the join back to RELATES_TO is dropped and the FTS IndexScan is
        # kept. Every EdgeDoc-column filter (group, edge name/uuid, temporal)
        # rides the PRE-bm25 WHERE, so the top-K is computed within the
        # filtered set (exact recall); the old shape filtered AFTER the LIMIT.
        doc_filters, doc_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG, edge_alias='d', emit_node_labels=False
        )
        if group_ids is not None:
            doc_filters.append(neug_in_filter('d.group_id', group_ids, doc_params))
        doc_prefilter = (' WHERE ' + ' AND '.join(doc_filters)) if doc_filters else ''

        if search_filter.node_labels:
            # node_labels is the one filter EdgeDoc cannot serve (no label
            # columns), so join to the endpoints and rank after the join. The
            # join runs BEFORE the top-K, so recall stays exact (bm25 still
            # resolves through the EdgeDoc FTS index).
            cypher = (
                """
                MATCH (d:EdgeDoc)
                """
                + doc_prefilter
                + """
                MATCH (n:Entity)-[e:RELATES_TO {uuid: d.uuid}]->(m:Entity)
                WHERE """
                + neug_node_label_filter(search_filter.node_labels)
                + """
                WITH d, n, m, e, bm25(d.fact, $query) AS score
                ORDER BY score ASC LIMIT """
                + str(int(limit))
                + """
                RETURN
                """
                + get_entity_edge_return_query(GraphProvider.NEUG)
                + """, score
                ORDER BY score ASC
                """
            )
        else:
            cypher = (
                """
                MATCH (d:EdgeDoc)
                """
                + doc_prefilter
                + """
                WITH d, bm25(d.fact, $query) AS score
                ORDER BY score ASC LIMIT """
                + str(int(limit))
                + """
                RETURN
                """
                + get_entity_edge_doc_return_query()
                + """, score
                ORDER BY score ASC
                """
            )

        records, _, _ = await executor.execute_query(
            cypher,
            query=fuzzy_query,
            **doc_params,
        )

        return [entity_edge_from_record(r) for r in records]

    async def edge_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityEdge]:
        doc_filters, doc_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG, edge_alias='d', emit_node_labels=False
        )
        if group_ids is not None:
            doc_filters.append(neug_in_filter('d.group_id', group_ids, doc_params))
            if source_node_uuid is not None:
                doc_params['source_uuid'] = source_node_uuid
                doc_filters.append('d.source_node_uuid = $source_uuid')
            if target_node_uuid is not None:
                doc_params['target_uuid'] = target_node_uuid
                doc_filters.append('d.target_node_uuid = $target_uuid')
        doc_prefilter = (' WHERE ' + ' AND '.join(doc_filters)) if doc_filters else ''

        # Rank over the EdgeDoc mirror and project the edge straight from it.
        # EdgeDoc carries every returned field plus both endpoint uuids, so the
        # join back to RELATES_TO is dropped: keeping vector_distance in the
        # TERMINAL `RETURN ... ORDER BY dist ASC LIMIT k` (no intervening WITH,
        # no join) is what lets the engine rewrite it into an HNSW IndexScan
        # with the WHERE pushed down as a scalar pre-filter. The old double-WITH
        # + post-LIMIT-join shape degraded to a brute-force scan AND ran the
        # group filter after the global top-K (a perf and a recall bug). Every
        # EdgeDoc-column filter rides the pre-RETURN WHERE. min_score trims the
        # ranked top-K client-side (monotone in dist). The query vector rides a
        # bound $search_vector; see node_search for why inlining it costs ~92%
        # of the query's latency in plan recompilation.
        if search_filter.node_labels:
            # node_labels is the one filter EdgeDoc cannot serve (no label
            # columns), so join to the endpoints and rank after the join. The
            # join defeats the HNSW rewrite (brute-force re-rank), but it runs
            # BEFORE the top-K, so recall stays exact.
            cypher = (
                """
                MATCH (d:EdgeDoc)
                """
                + doc_prefilter
                + """
                MATCH (n:Entity)-[e:RELATES_TO {uuid: d.uuid}]->(m:Entity)
                WHERE """
                + neug_node_label_filter(search_filter.node_labels)
                + """
                WITH d, n, m, e, vector_distance_cosine(d.fact_embedding, $search_vector) AS dist
                ORDER BY dist ASC LIMIT """
                + str(int(limit))
                + """
                RETURN
                """
                + get_entity_edge_return_query(GraphProvider.NEUG)
                + """, dist
                ORDER BY dist ASC
                """
            )
        else:
            cypher = (
                """
                MATCH (d:EdgeDoc)
                """
                + doc_prefilter
                + """
                RETURN
                """
                + get_entity_edge_doc_return_query()
                + """,
                vector_distance_cosine(d.fact_embedding, $search_vector) AS dist
                ORDER BY dist ASC LIMIT """
                + str(int(limit))
            )

        records, _, _ = await executor.execute_query(
            cypher,
            search_vector=search_vector,
            **doc_params,
        )

        return [entity_edge_from_record(r) for r in records if (1.0 - float(r['dist'])) > min_score]

    async def edge_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        if not origin_uuids or max_depth < 1:
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.NEUG
        )

        if group_ids is not None:
            filter_queries.append(neug_in_filter('e.group_id', group_ids, filter_params))

        # Each NEUG leg already carries a WHERE clause, so extra filters join
        # on with AND rather than introducing a second WHERE.
        neug_filter_query = ''
        if filter_queries:
            neug_filter_query = ' AND ' + (' AND '.join(filter_queries))

        # No UNWIND and no mixed-type traversal from an unlabelled origin, so
        # BFS runs as one leg per origin kind. An edge is within `depth` hops
        # iff its source node is reachable in <= depth - 1 hops; MENTIONS only
        # ever appears as the first hop out of an Episodic origin.
        origins = neug_in_filter('origin.uuid', origin_uuids, filter_params)
        match_queries = [
            f"""
            MATCH (origin:Entity)-[:RELATES_TO*0..{max_depth - 1}]->(n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE {origins}
            """,
        ]
        if max_depth >= 2:
            if max_depth == 2:
                episodic_leg = f"""
                MATCH (origin:Episodic)-[:MENTIONS]->(n:Entity)-[e:RELATES_TO]->(m:Entity)
                WHERE {origins}
                """
            else:
                episodic_leg = f"""
                MATCH (origin:Episodic)-[:MENTIONS]->(:Entity)-[:RELATES_TO*0..{max_depth - 2}]->(n:Entity)-[e:RELATES_TO]->(m:Entity)
                WHERE {origins}
                """
            match_queries.append(episodic_leg)

        records: list[dict[str, Any]] = []
        seen_uuids: set[str] = set()
        for match_query in match_queries:
            sub_records, _, _ = await executor.execute_query(
                match_query
                + neug_filter_query
                + """
                RETURN DISTINCT
                """
                + get_entity_edge_return_query(GraphProvider.NEUG)
                + """
                LIMIT """
                + str(int(limit)),
                **filter_params,
            )
            for record in sub_records:
                if record['uuid'] not in seen_uuids:
                    seen_uuids.add(record['uuid'])
                    records.append(record)

        return [entity_edge_from_record(r) for r in records]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,  # noqa: ARG002
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        fuzzy_query = dialect.build_fts_query(query)
        if fuzzy_query == '':
            return []

        # Single weighted multi-property bm25 over the Episodic(content,
        # source, source_description) FTS index (content weighted 3x). The
        # group filter rides the PRE-bm25 WHERE, so the top-K is computed
        # within the group (exact recall). Replaces three per-column queries
        # merged client-side with the filter applied AFTER ORDER BY/LIMIT.
        group_params: dict[str, Any] = {}
        group_filter = (
            ' WHERE ' + neug_in_filter('e.group_id', group_ids, group_params)
            if group_ids is not None
            else ''
        )
        cypher = (
            """
            MATCH (e:Episodic)
            """
            + group_filter
            + """
            WITH e, bm25([e.content, e.source, e.source_description], [3.0, 1.0, 1.0], $query) AS score
            ORDER BY score ASC LIMIT """
            + str(int(limit))
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(cypher, query=fuzzy_query, **group_params)

        return [episodic_node_from_record(r) for r in records]

    # --- Community search ---

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        fuzzy_query = dialect.build_fts_query(query)
        if fuzzy_query == '':
            return []

        params: dict[str, Any] = {'query': fuzzy_query}
        # bm25 over the Community(name) FTS index; the group filter rides the
        # PRE-bm25 WHERE, so the top-K is computed within the group (exact
        # recall) instead of AFTER ORDER BY/LIMIT.
        group_filter = (
            ' WHERE ' + neug_in_filter('c.group_id', group_ids, params)
            if group_ids is not None
            else ''
        )
        cypher = (
            """
            MATCH (c:Community)
            """
            + group_filter
            + """
            WITH c, bm25(c.name, $query) AS score
            ORDER BY score ASC LIMIT """
            + str(int(limit))
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score ASC
            """
        )

        records, _, _ = await executor.execute_query(cypher, **params)

        return [community_node_from_record(r) for r in records]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        # HNSW ANN path, same shape as node_similarity_search: a terminal
        #   RETURN ..., vector_distance_cosine(c.name_embedding, <lit>) AS dist
        #   ORDER BY dist ASC LIMIT k
        # is rewritten into an ANN IndexScan. The group filter rides the
        # pre-RETURN WHERE (pushed into the scan as a scalar pre-filter, so the
        # top-K is computed within the group - exact recall), and the extra
        # `WITH c, dist` projection is gone: it turned dist into a plain
        # variable reference the rewrite no longer recognizes, degrading to a
        # brute-force scan. min_score trims the ranked top-K client-side
        # (monotone in dist). The query vector rides a bound $search_vector;
        # see node_search for why inlining it costs ~92% of the query's latency
        # in plan recompilation.
        params: dict[str, Any] = {}
        group_filter = (
            ' WHERE ' + neug_in_filter('c.group_id', group_ids, params)
            if group_ids is not None
            else ''
        )
        cypher = (
            """
            MATCH (c:Community)
            """
            + group_filter
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """,
            vector_distance_cosine(c.name_embedding, $search_vector) AS dist
            ORDER BY dist ASC LIMIT """
            + str(int(limit))
        )

        records, _, _ = await executor.execute_query(
            cypher, search_vector=search_vector, **params
        )

        return [
            community_node_from_record(r) for r in records if (1.0 - float(r['dist'])) > min_score
        ]

    # --- Rerankers ---

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> list[EntityNode]:
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}

        # Skip the query entirely when nothing but the center remains.
        if len(filtered_uuids) > 0:
            params: dict[str, Any] = {'center_uuid': center_node_uuid}
            results, _, _ = await executor.execute_query(
                """
                MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity)
                WHERE """
                + neug_in_filter('n.uuid', filtered_uuids, params)
                + """
                RETURN 1 AS score, n.uuid AS uuid
                """,
                **params,
            )
            for result in results:
                scores[result['uuid']] = result['score']

        for uuid in filtered_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        # rerank on shortest distance
        filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        # add back in filtered center uuid if it was filtered out
        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        reranked_uuids = [u for u in filtered_uuids if (1 / scores[u]) >= min_score]

        return await self._get_entity_nodes_by_uuids(executor, reranked_uuids)

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        min_score: float = 0,
    ) -> list[EntityNode]:
        if not node_uuids:
            return []

        scores: dict[str, float] = {}

        params: dict[str, Any] = {}
        results, _, _ = await executor.execute_query(
            """
            MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity)
            WHERE """
            + neug_in_filter('n.uuid', node_uuids, params)
            + """
            RETURN count(*) AS score, n.uuid AS uuid
            """,
            **params,
        )
        for result in results:
            scores[result['uuid']] = result['score']

        for uuid in node_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        sorted_uuids = list(node_uuids)
        sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        reranked_uuids = [u for u in sorted_uuids if scores[u] >= min_score]

        return await self._get_entity_nodes_by_uuids(executor, reranked_uuids)

    async def _get_entity_nodes_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityNode]:
        if not uuids:
            return []

        params: dict[str, Any] = {}
        records, _, _ = await executor.execute_query(
            """
            MATCH (n:Entity)
            WHERE """
            + neug_in_filter('n.uuid', uuids, params)
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEUG),
            **params,
        )

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
        return [node_map[u] for u in uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filters, GraphProvider.NEUG
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filters, GraphProvider.NEUG
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,  # noqa: ARG002
        max_query_length: int = 8000,
    ) -> str:
        # NeuG's bm25() takes a plain term query; group filtering is applied
        # separately on the matched rows.
        return dialect.build_fts_query(query, max_query_length)
