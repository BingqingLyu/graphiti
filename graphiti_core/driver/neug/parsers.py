"""
Record parsers for the NeuG backend.

These convert the plain-dict rows produced by ``NeuGDriver.execute_query``
into Graphiti node/edge model instances. They mirror the field sets of the
canonical ``get_*_from_record`` helpers in ``graphiti_core.nodes`` and
``graphiti_core.edges`` (the Kuzu-style variants), because NeuG — like Kuzu —
stores list/dict attributes in dedicated JSON string columns rather than
spreading them across the node/edge property map.

Two NeuG-specific coercions applied everywhere:

- Datetimes come back as fixed-width ISO-8601 UTC strings (see
  ``dialect.parse_iso``); unset values arrive as ``''`` and parse to ``None``.
- List columns (labels, entity_edges, episodes) are native STRING[] columns
  and arrive as Python lists; dict columns (attributes, episode_metadata) are
  JSON string columns decoded via ``dialect.from_json_col``. Unset values
  arrive as ``''`` and fall back to the field's default.
"""

from __future__ import annotations

from typing import Any

from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
)

from .dialect import from_json_col, parse_iso


def str_list_col(value: Any, default: list[str]) -> list[str]:
    """Read a STRING[] column; unset columns come back as ''."""
    if value is None or value == '':
        return default
    if isinstance(value, list):
        return value
    return from_json_col(value, default)


def entity_node_from_record(record: dict[str, Any]) -> EntityNode:
    labels = str_list_col(record.get('labels'), [])
    # Defensive: strip the synthetic per-group label if it ever round-trips.
    group_id = record.get('group_id') or ''
    synthetic = 'Entity_' + group_id.replace('-', '')
    if synthetic in labels:
        labels = [label for label in labels if label != synthetic]

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
        summary=record.get('summary') or '',
        attributes=from_json_col(record.get('attributes'), {}),
    )


def episodic_node_from_record(record: dict[str, Any]) -> EpisodicNode:
    created_at = parse_iso(record.get('created_at'))
    valid_at = parse_iso(record.get('valid_at'))

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record.get('content') or '',
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record.get('group_id') or '',
        source=EpisodeType.from_str(record.get('source') or 'text'),
        name=record.get('name') or '',
        source_description=record.get('source_description') or '',
        entity_edges=str_list_col(record.get('entity_edges'), []),
        episode_metadata=from_json_col(record.get('episode_metadata'), None),
    )


def community_node_from_record(record: dict[str, Any]) -> CommunityNode:
    return CommunityNode(
        uuid=record['uuid'],
        name=record.get('name') or '',
        group_id=record.get('group_id') or '',
        name_embedding=record.get('name_embedding'),
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
        summary=record.get('summary') or '',
    )


def saga_node_from_record(record: dict[str, Any]) -> SagaNode:
    return SagaNode(
        uuid=record['uuid'],
        name=record.get('name') or '',
        group_id=record.get('group_id') or '',
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
        summary=record.get('summary') or '',
        first_episode_uuid=record.get('first_episode_uuid') or None,
        last_episode_uuid=record.get('last_episode_uuid') or None,
        last_summarized_at=parse_iso(record.get('last_summarized_at')),
        last_summarized_episode_valid_at=parse_iso(record.get('last_summarized_episode_valid_at')),
    )


def episodic_edge_from_record(record: dict[str, Any]) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id') or '',
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
    )


def entity_edge_from_record(record: dict[str, Any]) -> EntityEdge:
    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record.get('fact') or '',
        fact_embedding=record.get('fact_embedding'),
        name=record.get('name') or '',
        group_id=record.get('group_id') or '',
        episodes=str_list_col(record.get('episodes'), []),
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
        expired_at=parse_iso(record.get('expired_at')),
        valid_at=parse_iso(record.get('valid_at')),
        invalid_at=parse_iso(record.get('invalid_at')),
        reference_time=parse_iso(record.get('reference_time')),
        attributes=from_json_col(record.get('attributes'), {}),
    )


def community_edge_from_record(record: dict[str, Any]) -> CommunityEdge:
    return CommunityEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id') or '',
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
    )


def has_episode_edge_from_record(record: dict[str, Any]) -> HasEpisodeEdge:
    return HasEpisodeEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id') or '',
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
    )


def next_episode_edge_from_record(record: dict[str, Any]) -> NextEpisodeEdge:
    return NextEpisodeEdge(
        uuid=record['uuid'],
        group_id=record.get('group_id') or '',
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_iso(record.get('created_at')),  # type: ignore[arg-type]
    )
