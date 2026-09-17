"""
EdgeDoc mirror maintenance for the NeuG backend.

NeuG can only build FTS/HNSW indexes on node tables, so edge-level retrieval
runs against the `EdgeDoc` mirror node table (which shares `uuid` with the
real `RELATES_TO` edge). Every code path that writes or deletes an entity edge
must also apply the corresponding mirror operation; the statements live here
so all call sites stay in sync.
"""

from __future__ import annotations

from typing import Any

NEUG_EDGE_DOC_UPSERT = """
    MERGE (d:EdgeDoc {uuid: $uuid})
    SET
        d.source_node_uuid = $source_node_uuid,
        d.target_node_uuid = $target_node_uuid,
        d.group_id = $group_id,
        d.name = $name,
        d.fact = $fact,
        d.fact_embedding = $fact_embedding,
        d.episodes = $episodes,
        d.attributes = $attributes,
        d.created_at = $created_at,
        d.expired_at = $expired_at,
        d.valid_at = $valid_at,
        d.invalid_at = $invalid_at,
        d.reference_time = $reference_time
    RETURN d.uuid AS uuid
"""

NEUG_EDGE_DOC_DELETE = 'MATCH (d:EdgeDoc {uuid: $uuid}) DETACH DELETE d;'


def edge_doc_upsert_params(edge_data: dict[str, Any]) -> dict[str, Any]:
    """Project an entity-edge dict onto the EdgeDoc mirror columns.

    Accepts the same dict shape that the bulk/entity-edge save paths use
    (Kuzu-style: datetimes still as datetime objects, `attributes` already
    JSON-serialized).
    """
    return {
        'uuid': edge_data['uuid'],
        'source_node_uuid': edge_data['source_node_uuid'],
        'target_node_uuid': edge_data['target_node_uuid'],
        'group_id': edge_data.get('group_id'),
        'name': edge_data.get('name'),
        'fact': edge_data.get('fact'),
        'fact_embedding': edge_data.get('fact_embedding'),
        'episodes': edge_data.get('episodes'),
        'attributes': edge_data.get('attributes'),
        'created_at': edge_data.get('created_at'),
        'expired_at': edge_data.get('expired_at'),
        'valid_at': edge_data.get('valid_at'),
        'invalid_at': edge_data.get('invalid_at'),
        'reference_time': edge_data.get('reference_time'),
    }
