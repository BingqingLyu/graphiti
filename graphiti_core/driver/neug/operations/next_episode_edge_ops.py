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

from graphiti_core.driver.neug.dialect import neug_in_filter, neug_limit
from graphiti_core.driver.neug.parsers import next_episode_edge_from_record
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import NextEpisodeEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.models.edges.edge_db_queries import (
    NEXT_EPISODE_EDGE_RETURN,
    NEXT_EPISODE_EDGE_SAVE,
)

logger = logging.getLogger(__name__)


class NeuGNextEpisodeEdgeOperations(NextEpisodeEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        params: dict[str, Any] = {
            'source_episode_uuid': edge.source_node_uuid,
            'target_episode_uuid': edge.target_node_uuid,
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'created_at': edge.created_at,
        }
        if tx is not None:
            await tx.run(NEXT_EPISODE_EDGE_SAVE, **params)
        else:
            await executor.execute_query(NEXT_EPISODE_EDGE_SAVE, **params)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[NextEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        # NeuG cannot UNWIND bound lists - iterate and save individually
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: NextEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n:Episodic)-[e:NEXT_EPISODE {uuid: $uuid}]->(m:Episodic)
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuid=edge.uuid)
        else:
            await executor.execute_query(query, uuid=edge.uuid)

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
        query = (
            """
            MATCH (n:Episodic)-[e:NEXT_EPISODE]->(m:Episodic)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            DELETE e
            """
        )
        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> NextEpisodeEdge:
        query = (
            """
            MATCH (n:Episodic)-[e:NEXT_EPISODE {uuid: $uuid}]->(m:Episodic)
            RETURN
            """
            + NEXT_EPISODE_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [next_episode_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[NextEpisodeEdge]:
        params: dict[str, Any] = {}
        query = (
            """
            MATCH (n:Episodic)-[e:NEXT_EPISODE]->(m:Episodic)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            RETURN
            """
            + NEXT_EPISODE_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [next_episode_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[NextEpisodeEdge]:
        params: dict[str, Any] = {'uuid': uuid_cursor}
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        query = (
            """
            MATCH (n:Episodic)-[e:NEXT_EPISODE]->(m:Episodic)
            WHERE """
            + neug_in_filter('e.group_id', group_ids, params)
            + '\n'
            + cursor_clause
            + """
            RETURN
            """
            + NEXT_EPISODE_EDGE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + neug_limit(limit)
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [next_episode_edge_from_record(r) for r in records]
