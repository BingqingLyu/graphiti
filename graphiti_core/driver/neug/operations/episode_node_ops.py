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
from datetime import datetime
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neug.dialect import neug_in_filter, neug_limit
from graphiti_core.driver.neug.parsers import episodic_node_from_record
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    get_episode_node_save_query,
)
from graphiti_core.nodes import EpisodicNode

logger = logging.getLogger(__name__)


class NeuGEpisodeNodeOperations(EpisodeNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        params: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'source_description': node.source_description,
            'content': node.content,
            'entity_edges': node.entity_edges,
            'created_at': node.created_at,
            'valid_at': node.valid_at,
            'source': node.source.value,
        }

        query = get_episode_node_save_query(GraphProvider.NEUG)

        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EpisodicNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        # NeuG cannot UNWIND bound lists - iterate and save individually
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (e:Episodic {uuid: $uuid})
            DETACH DELETE e
        """
        if tx is not None:
            await tx.run(query, uuid=node.uuid)
        else:
            await executor.execute_query(query, uuid=node.uuid)

        logger.debug(f'Deleted Node: {node.uuid}')

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        query = """
            MATCH (e:Episodic {group_id: $group_id})
            DETACH DELETE e
        """
        if tx is not None:
            await tx.run(query, group_id=group_id)
        else:
            await executor.execute_query(query, group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        if not uuids:
            return
        params: dict[str, Any] = {}
        query = (
            """
            MATCH (e:Episodic)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            DETACH DELETE e
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
    ) -> EpisodicNode:
        query = (
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        nodes = [episodic_node_from_record(r) for r in records]
        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EpisodicNode]:
        params: dict[str, Any] = {}
        query = (
            """
            MATCH (e:Episodic)
            WHERE """
            + neug_in_filter('e.uuid', uuids, params)
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [episodic_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        params: dict[str, Any] = {'uuid': uuid_cursor}
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        query = (
            """
            MATCH (e:Episodic)
            WHERE """
            + neug_in_filter('e.group_id', group_ids, params)
            + '\n'
            + cursor_clause
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + neug_limit(limit)
        )
        records, _, _ = await executor.execute_query(query, **params)
        return [episodic_node_from_record(r) for r in records]

    async def get_by_entity_node_uuid(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        query = (
            """
            MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, entity_node_uuid=entity_node_uuid)
        return [episodic_node_from_record(r) for r in records]

    async def retrieve_episodes(
        self,
        executor: QueryExecutor,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        if saga is not None:
            group_id = group_ids[0] if group_ids else None
            source_filter = 'AND e.source = $source' if source is not None else ''

            records, _, _ = await executor.execute_query(
                f"""
                MATCH (s:Saga {{name: $saga_name, group_id: $group_id}})-[:HAS_EPISODE]->(e:Episodic)
                WHERE e.valid_at <= $reference_time
                {source_filter}
                RETURN
                """
                + EPISODIC_NODE_RETURN
                # NeuG only accepts literal LIMITs.
                + f"""
                ORDER BY e.valid_at DESC
                LIMIT {int(last_n)}
                """,
                saga_name=saga,
                group_id=group_id,
                reference_time=reference_time,
                source=source,
            )

            episodes = [episodic_node_from_record(r) for r in records]
            return list(reversed(episodes))  # Return in chronological order

        query_params: dict[str, Any] = {'reference_time': reference_time}
        query_filter = ''
        if group_ids:
            query_filter += '\nAND ' + neug_in_filter('e.group_id', group_ids, query_params)
        if source is not None:
            query_filter += '\nAND e.source = $source'
            query_params['source'] = source

        query = (
            """
            MATCH (e:Episodic)
            WHERE e.valid_at <= $reference_time
            """
            + query_filter
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + f"""
            ORDER BY e.valid_at DESC
            LIMIT {int(last_n)}
            """
        )
        records, _, _ = await executor.execute_query(query, **query_params)

        episodes = [episodic_node_from_record(r) for r in records]
        return list(reversed(episodes))  # Return in chronological order
