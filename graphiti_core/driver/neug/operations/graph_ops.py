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

from graphiti_core.driver.driver import (
    COMMUNITY_INDEX_NAME,
    ENTITY_EDGE_INDEX_NAME,
    ENTITY_INDEX_NAME,
    EPISODE_INDEX_NAME,
    GraphProvider,
)
from graphiti_core.driver.neug import schema
from graphiti_core.driver.neug.dialect import neug_in_filter
from graphiti_core.driver.neug.parsers import community_node_from_record, entity_node_from_record
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.helpers import semaphore_gather
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)


class NeuGGraphMaintenanceOperations(GraphMaintenanceOperations):
    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids is None:
            await executor.execute_query('MATCH (n) DETACH DELETE n')
        else:
            # EdgeDoc mirrors carry edge data; Saga holds episode chains, so
            # they must be cleared alongside the three standard node labels.
            params: dict[str, Any] = {}
            group_filter = neug_in_filter('n.group_id', group_ids, params)
            for label in ['Entity', 'Episodic', 'Community', 'Saga', 'EdgeDoc']:
                await executor.execute_query(
                    f"""
                    MATCH (n:{label})
                    WHERE {group_filter}
                    DETACH DELETE n
                    """,
                    **params,
                )

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        if delete_existing:
            await self.delete_all_indexes(executor)

        index_queries = schema.index_ddl(
            ENTITY_INDEX_NAME, EPISODE_INDEX_NAME, COMMUNITY_INDEX_NAME, ENTITY_EDGE_INDEX_NAME
        )
        await semaphore_gather(*[executor.execute_query(q) for q in index_queries])

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        drop_queries = schema.index_drop_ddl(
            ENTITY_INDEX_NAME, EPISODE_INDEX_NAME, COMMUNITY_INDEX_NAME, ENTITY_EDGE_INDEX_NAME
        )
        await semaphore_gather(*[executor.execute_query(q) for q in drop_queries])

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        community_clusters: list[list[EntityNode]] = []

        if group_ids is None:
            # One row per distinct group_id; the driver gathers them here.
            group_id_values, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id IS NOT NULL
                RETURN DISTINCT n.group_id AS group_id
                """
            )
            group_ids = [record['group_id'] for record in group_id_values if record['group_id']]

        resolved_group_ids: list[str] = group_ids or []
        for group_id in resolved_group_ids:
            # Get all entity nodes for this group
            params: dict[str, Any] = {'group_id': group_id}
            node_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE """
                + neug_in_filter('n.group_id', [group_id], params)
                + """
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.NEUG),
                **params,
            )
            nodes = [entity_node_from_record(r) for r in node_records]

            # One grouped query returns every (node, neighbor, edge_count)
            # triple for the whole group — NeuG supports this aggregation, so
            # the previous per-node loop (an N+1: one query per entity) is
            # collapsed into a single round-trip. Seed every node with an empty
            # adjacency first so isolated nodes still appear in the projection.
            projection: dict[str, list[Neighbor]] = {node.uuid: [] for node in nodes}
            neighbor_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]-(m:Entity {group_id: $group_id})
                RETURN n.uuid AS node_uuid, m.uuid AS uuid, count(e) AS count
                """,
                group_id=group_id,
            )
            for record in neighbor_records:
                neighbors = projection.get(record['node_uuid'])
                if neighbors is not None:
                    neighbors.append(Neighbor(node_uuid=record['uuid'], edge_count=record['count']))

            cluster_uuids = label_propagation(projection)

            # Fetch full node objects for each cluster
            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_params: dict[str, Any] = {}
                cluster_records, _, _ = await executor.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE """
                    + neug_in_filter('n.uuid', cluster, cluster_params)
                    + """
                    RETURN
                    """
                    + get_entity_node_return_query(GraphProvider.NEUG),
                    **cluster_params,
                )
                community_clusters.append([entity_node_from_record(r) for r in cluster_records])

        return community_clusters

    # remove_communities is not overridden: the base implementation binds
    # `IN $group_ids`, supported by the current engine build.

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        # Check if the node is already part of a community
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

        if len(records) > 0:
            return

        # If the node has no community, find the mode community of surrounding entities.
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[e:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        episode_uuids = [episode.uuid for episode in episodes]
        if not episode_uuids:
            return []

        params: dict[str, Any] = {}
        records, _, _ = await executor.execute_query(
            """
            MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
            WHERE """
            + neug_in_filter('episode.uuid', episode_uuids, params)
            + """
            RETURN DISTINCT
            """
            + get_entity_node_return_query(GraphProvider.NEUG),
            **params,
        )

        return [entity_node_from_record(r) for r in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        node_uuids = [node.uuid for node in nodes]
        if not node_uuids:
            return []

        params: dict[str, Any] = {}
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)
            WHERE """
            + neug_in_filter('m.uuid', node_uuids, params)
            + """
            RETURN DISTINCT
            """
            + COMMUNITY_NODE_RETURN,
            **params,
        )

        return [community_node_from_record(r) for r in records]
