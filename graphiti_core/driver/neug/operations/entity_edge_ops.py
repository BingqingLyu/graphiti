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

import json
import logging
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neug.dialect import neug_in_filter, neug_limit
from graphiti_core.driver.neug.edge_mirror import (
    NEUG_EDGE_DOC_DELETE,
    NEUG_EDGE_DOC_UPSERT,
    edge_doc_upsert_params,
)
from graphiti_core.driver.neug.parsers import entity_edge_from_record
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_return_query,
    get_entity_edge_save_query,
)

logger = logging.getLogger(__name__)


class NeuGEntityEdgeOperations(EntityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        # NeuG binds flat $params (nested map binding is unsupported);
        # attributes are JSON-serialized into their STRING column.
        params: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_uuid': edge.source_node_uuid,
            'target_uuid': edge.target_node_uuid,
            'name': edge.name,
            'group_id': edge.group_id,
            'fact': edge.fact,
            'fact_embedding': edge.fact_embedding,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'reference_time': edge.reference_time,
            'attributes': json.dumps(edge.attributes or {}),
        }

        query = get_entity_edge_save_query(GraphProvider.NEUG)

        # The EdgeDoc mirror (edge-level retrieval index) must stay in sync
        # with the real RELATES_TO edge.
        mirror_params = edge_doc_upsert_params(
            {
                'uuid': edge.uuid,
                'source_node_uuid': edge.source_node_uuid,
                'target_node_uuid': edge.target_node_uuid,
                'group_id': edge.group_id,
                'name': edge.name,
                'fact': edge.fact,
                'fact_embedding': edge.fact_embedding,
                'episodes': edge.episodes,
                'attributes': json.dumps(edge.attributes or {}),
                'created_at': edge.created_at,
                'expired_at': edge.expired_at,
                'valid_at': edge.valid_at,
                'invalid_at': edge.invalid_at,
                'reference_time': edge.reference_time,
            }
        )

        if tx is not None:
            await tx.run(query, **params)
            await tx.run(NEUG_EDGE_DOC_UPSERT, **mirror_params)
        else:
            await executor.execute_query(query, **params)
            await executor.execute_query(NEUG_EDGE_DOC_UPSERT, **mirror_params)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        # NeuG binds `UNWIND $list` but cannot execute it (verified on 0.2.0:
        # scalar-list unfold throws, map-list serialization fails, and
        # `UNWIND $ids ... MERGE` fails pipeline construction), so bulk saves
        # iterate the per-edge MERGE. Engine limitation, not a Cypher-authoring
        # mistake — no UNWIND form of this save works today.
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        delete_query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            DELETE e
        """
        if tx is not None:
            await tx.run(delete_query, uuid=edge.uuid)
            await tx.run(NEUG_EDGE_DOC_DELETE, uuid=edge.uuid)
        else:
            await executor.execute_query(delete_query, uuid=edge.uuid)
            await executor.execute_query(NEUG_EDGE_DOC_DELETE, uuid=edge.uuid)

        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        if not uuids:
            return
        params: dict[str, Any] = {}
        uuids_filter = neug_in_filter('e.uuid', uuids, params)
        delete_query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE """
            + uuids_filter
            + """
            DELETE e
            """
        )
        mirror_query = (
            """
            MATCH (d:EdgeDoc)
            WHERE """
            + neug_in_filter('d.uuid', uuids, params)
            + """
            DETACH DELETE d
            """
        )
        if tx is not None:
            await tx.run(delete_query, **params)
            await tx.run(mirror_query, **params)
        else:
            await executor.execute_query(delete_query, **params)
            await executor.execute_query(mirror_query, **params)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityEdge:
        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEUG)
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [entity_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityEdge]:
        if len(uuids) == 0:
            return []
        params: dict[str, Any] = {}
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.NEUG)
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [entity_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        params: dict[str, Any] = {'uuid': uuid_cursor}
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE """
            + neug_in_filter('e.group_id', group_ids, params)
            + '\n'
            + cursor_clause
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.NEUG)
            + """
            ORDER BY e.uuid DESC
            """
            + neug_limit(limit)
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [entity_edge_from_record(r) for r in records]

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        query = """
            MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEUG)
        records, _, _ = await executor.execute_query(
            query,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
        )
        return [entity_edge_from_record(r) for r in records]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        # NeuG names the endpoint functions START_NODE()/END_NODE() and does
        # not allow extracting properties from them directly, so they are
        # bound in a WITH layer; the undirected match then always binds n to
        # the source and m to the target.
        query = """
            MATCH (x:Entity {uuid: $node_uuid})-[e:RELATES_TO]-()
            WITH e, START_NODE(e) AS n, END_NODE(e) AS m
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEUG)
        records, _, _ = await executor.execute_query(query, node_uuid=node_uuid)
        return [entity_edge_from_record(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            RETURN e.fact_embedding AS fact_embedding
        """
        records, _, _ = await executor.execute_query(query, uuid=edge.uuid)
        if len(records) == 0:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = records[0]['fact_embedding']

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        if not edges:
            return
        uuids = [e.uuid for e in edges]
        params: dict[str, Any] = {}
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            RETURN DISTINCT e.uuid AS uuid, e.fact_embedding AS fact_embedding
            """
        )
        records, _, _ = await executor.execute_query(query, **params)
        embedding_map = {r['uuid']: r['fact_embedding'] for r in records}
        for edge in edges:
            if edge.uuid in embedding_map:
                edge.fact_embedding = embedding_map[edge.uuid]
