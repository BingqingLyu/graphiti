"""
Bulk ingestion for the NeuG backend via COPY FROM CSV.

The generic bulk path issues one MERGE statement per node/edge, which on
NeuG means one embedded-engine round trip per row. COPY FROM ingests a whole
CSV in a single statement and is orders of magnitude faster (measured: 10k
nodes + 10k edges in ~1s vs minutes for per-row MERGE).

Semantics preserved relative to the MERGE path:

- Fresh rows (episodic nodes, entity edges, EdgeDoc mirrors, MENTIONS edges,
  and entity nodes not yet in the graph) are COPYed. COPY silently skips
  primary-key collisions, so existing rows are never clobbered.
- COPY demands a CSV column set matching the full table schema (it errors
  on a sniffed-schema mismatch), so every column of the target table
  appears in each file.
- Entity nodes whose uuid already exists still need their properties
  refreshed (dedupe re-saves canonical nodes), so those fall back to the
  per-row MERGE — a small minority of each batch.
- Nullable temporal STRING columns are written as '' (the driver's null
  convention; reads normalize '' back to None), which also sidesteps the
  engine's rejection of SET NULL on edge updates.
"""

from __future__ import annotations

import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neug import dialect
from graphiti_core.driver.neug.edge_mirror import NEUG_EDGE_DOC_UPSERT, edge_doc_upsert_params
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_bulk_query

if TYPE_CHECKING:
    from graphiti_core.driver.driver import GraphDriver


def _cell(value: Any) -> str:
    """Render one value for a CSV cell under the driver's storage conventions."""
    if value is None:
        return ''
    if isinstance(value, datetime):
        return dialect.iso(value)
    if isinstance(value, dict):
        # dict-valued columns (episode_metadata, attributes) are JSON strings
        return dialect.json_col(value)
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)):
            # Embeddings are L2-normalized before writing, matching the
            # driver's ``cosine_normalize = false`` HNSW indexes.
            return dialect.vector_literal(dialect.l2_normalize(value))
        return '[' + ','.join(str(v) for v in value) + ']'
    return str(value)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(col)) for col in columns])


async def _copy(driver: GraphDriver, table: str, csv_path: Path):
    # ESCAPE="" turns off CSV escape parsing: without it NeuG silently strips
    # backslashes inside fields (probe-verified on the benchmark build), which
    # mangles node text/summary and can drop edges whose keys contain them.
    await driver.execute_query(f'COPY {table} FROM "{csv_path}" (HEADER=true, DELIMITER=",", ESCAPE="")')


async def _existing_entity_uuids(driver: GraphDriver, uuids: list[str]) -> set[str]:
    """Which of these entity uuids are already in the graph (single query)."""
    if not uuids:
        return set()
    params: dict[str, Any] = {}
    rows, _, _ = await driver.execute_query(
        f'MATCH (n:Entity) WHERE {dialect.neug_in_filter("n.uuid", uuids, params)} '
        'RETURN n.uuid AS uuid',
        **params,
    )
    return {row['uuid'] for row in rows}


async def _existing_edge_uuids(driver: GraphDriver, uuids: list[str]) -> set[str]:
    """Which of these entity-edge uuids are already in the graph (single query).

    Edges reaching the bulk path already present in the graph are the ones
    being invalidated or dedupe-updated; COPY would silently skip them, so
    they must take the MERGE path instead.
    """
    if not uuids:
        return set()
    params: dict[str, Any] = {}
    rows, _, _ = await driver.execute_query(
        f'MATCH ()-[r:RELATES_TO]->() WHERE {dialect.neug_in_filter("r.uuid", uuids, params)} '
        'RETURN r.uuid AS uuid',
        **params,
    )
    return {row['uuid'] for row in rows}


EPISODIC_COLUMNS = [
    'uuid',
    'name',
    'group_id',
    'source',
    'source_description',
    'content',
    'entity_edges',
    'episode_metadata',
    'created_at',
    'valid_at',
]
ENTITY_COLUMNS = [
    'uuid',
    'name',
    'name_embedding',
    'group_id',
    'summary',
    'labels',
    'attributes',
    'created_at',
]
RELATES_TO_COLUMNS = [
    'from',
    'to',
    'uuid',
    'name',
    'fact',
    'fact_embedding',
    'episodes',
    'group_id',
    'attributes',
    'created_at',
    'expired_at',
    'valid_at',
    'invalid_at',
    'reference_time',
]
EDGEDOC_COLUMNS = [
    'uuid',
    'source_node_uuid',
    'target_node_uuid',
    'name',
    'fact',
    'fact_embedding',
    'episodes',
    'group_id',
    'attributes',
    'created_at',
    'expired_at',
    'valid_at',
    'invalid_at',
    'reference_time',
]
MENTIONS_COLUMNS = ['from', 'to', 'uuid', 'group_id', 'created_at']


async def neug_bulk_copy_write(
    driver: GraphDriver,
    episodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    entity_edges: list[dict[str, Any]],
    episodic_edges: list[dict[str, Any]],
    entity_node_merge_query: str,
):
    """Persist one bulk batch via COPY FROM, MERGE-updating existing rows.

    Row dicts carry the same shapes the Kuzu/NeuG MERGE path consumes:
    datetimes as datetime objects, ``attributes`` already JSON-serialized,
    embeddings as float lists.

    Rows whose primary key already exists in the graph (entity nodes the
    dedupe re-saved as canonical, entity edges being invalidated or refreshed)
    take the per-row MERGE path because COPY silently skips collisions
    instead of updating them.
    """
    with tempfile.TemporaryDirectory(prefix='graphiti_neug_bulk_') as tmp:
        tmp_path = Path(tmp)

        if entity_nodes:
            existing_nodes = await _existing_entity_uuids(
                driver, [node['uuid'] for node in entity_nodes]
            )
            fresh_nodes = [n for n in entity_nodes if n['uuid'] not in existing_nodes]
            stale_nodes = [n for n in entity_nodes if n['uuid'] in existing_nodes]
        else:
            fresh_nodes, stale_nodes = [], []

        if entity_edges:
            existing_edges = await _existing_edge_uuids(
                driver, [edge['uuid'] for edge in entity_edges]
            )
            fresh_edges = [e for e in entity_edges if e['uuid'] not in existing_edges]
            stale_edges = [e for e in entity_edges if e['uuid'] in existing_edges]
        else:
            fresh_edges, stale_edges = [], []

        if episodes:
            p = tmp_path / 'episodic.csv'
            _write_csv(p, EPISODIC_COLUMNS, episodes)
            await _copy(driver, 'Episodic', p)
        if fresh_nodes:
            p = tmp_path / 'entity.csv'
            _write_csv(p, ENTITY_COLUMNS, fresh_nodes)
            await _copy(driver, 'Entity', p)
        if fresh_edges:
            rel_rows = [
                {'from': e['source_node_uuid'], 'to': e['target_node_uuid'], **e}
                for e in fresh_edges
            ]
            p = tmp_path / 'relates_to.csv'
            _write_csv(p, RELATES_TO_COLUMNS, rel_rows)
            await _copy(driver, 'RELATES_TO', p)

            doc_rows = [edge_doc_upsert_params(e) for e in fresh_edges]
            p = tmp_path / 'edgedoc.csv'
            _write_csv(p, EDGEDOC_COLUMNS, doc_rows)
            await _copy(driver, 'EdgeDoc', p)
        if episodic_edges:
            mention_rows = [
                {'from': e['source_node_uuid'], 'to': e['target_node_uuid'], **e}
                for e in episodic_edges
            ]
            p = tmp_path / 'mentions.csv'
            _write_csv(p, MENTIONS_COLUMNS, mention_rows)
            await _copy(driver, 'MENTIONS', p)

        # MERGE-updates for rows COPY skipped. Few per batch (only dedupe
        # hits and invalidations), so per-row statements are acceptable.
        if stale_edges:
            edge_merge_query = get_entity_edge_save_bulk_query(GraphProvider.NEUG)
            for edge in stale_edges:
                await driver.execute_query(edge_merge_query, **edge)
                # Keep the EdgeDoc mirror (edge FTS/HNSW indexes) in sync.
                await driver.execute_query(NEUG_EDGE_DOC_UPSERT, **edge_doc_upsert_params(edge))
        for node in stale_nodes:
            await driver.execute_query(entity_node_merge_query, **node)
