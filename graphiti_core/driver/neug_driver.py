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

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import neug
else:
    try:
        import neug
    except ImportError:
        raise ImportError(
            'neug is required for NeuGDriver. Install it with: pip install graphiti-core[neug]'
        ) from None

from graphiti_core.driver.driver import (
    COMMUNITY_INDEX_NAME,
    ENTITY_EDGE_INDEX_NAME,
    ENTITY_INDEX_NAME,
    EPISODE_INDEX_NAME,
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.driver.neug import dialect, schema
from graphiti_core.driver.neug.operations.community_edge_ops import NeuGCommunityEdgeOperations
from graphiti_core.driver.neug.operations.community_node_ops import NeuGCommunityNodeOperations
from graphiti_core.driver.neug.operations.entity_edge_ops import NeuGEntityEdgeOperations
from graphiti_core.driver.neug.operations.entity_node_ops import NeuGEntityNodeOperations
from graphiti_core.driver.neug.operations.episode_node_ops import NeuGEpisodeNodeOperations
from graphiti_core.driver.neug.operations.episodic_edge_ops import NeuGEpisodicEdgeOperations
from graphiti_core.driver.neug.operations.graph_ops import NeuGGraphMaintenanceOperations
from graphiti_core.driver.neug.operations.has_episode_edge_ops import NeuGHasEpisodeEdgeOperations
from graphiti_core.driver.neug.operations.next_episode_edge_ops import (
    NeuGNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neug.operations.saga_node_ops import NeuGSagaNodeOperations
from graphiti_core.driver.neug.operations.search_ops import NeuGSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.embedder.client import EMBEDDING_DIM

logger = logging.getLogger(__name__)


def _records_of(result: Any) -> list[dict[str, Any]]:
    """Materialize a NeuG result object into a list of row dicts.

    Uses O(n) row iteration (``list(result)`` + ``column_names()``) keyed by the
    RETURN alias (queries should alias every returned column). This replaced a
    ``get_bolt_response()`` parse: the bolt path serializes the whole result set
    into one JSON blob and is O(n^2) in the returned row count, which made large
    varlen graph traversals (tens of thousands of rows) take tens of seconds.
    Row iteration returns byte-identical rows: measured ~282x faster on such a
    traversal, with the materialized rows comparing equal field for field.
    Write/DDL statements yield no columns and return [].

    Temporal fields normalize ``''`` to None: NeuG reads unset STRING columns
    back as ``''`` rather than NULL, and all nullable columns in the schema are
    datetimes (``*_at`` / ``*_time``), which Graphiti parses via
    ``parse_db_date`` and expects as None when absent.
    """
    try:
        raw_rows = list(result)
    except Exception:
        raw_rows = []
    try:
        keys = list(result.column_names())
    except Exception:
        keys = []
    if not keys:
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(zip(keys, list(raw)))
        rows.append(
            {
                k: (None if v == '' and (k.endswith('_at') or k.endswith('_time')) else v)
                for k, v in row.items()
            }
        )
    return rows


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Adapt Python values to what the NeuG binding accepts.

    Datetimes become fixed-width ISO-8601 UTC strings (the storage convention
    used throughout this driver, so lexicographic comparison preserves order).
    None becomes '' — recent engine builds reject `SET col = NULL` on edge
    updates (batch_update_edge.cc), and unset STRING properties read back as
    '' anyway, so '' is the driver's established empty convention.
    Float lists are embeddings: they are L2-normalized before writing, the
    counterpart of the driver's ``cosine_normalize = false`` HNSW indexes
    (see dialect.l2_normalize). Int lists are left untouched — they are
    never embeddings and may be bound `IN $list` parameters on recent
    engines.

    Float elements are coerced to Python ``float``. The binding rejects numpy
    scalars with "Unsupported parameter type for serialization" and callers
    hand us ``list(np.ndarray)``, whose elements are ``np.float32``. Doing the
    coercion here is what lets search queries bind ``$search_vector`` instead
    of inlining the vector into the query text — inlining makes every query
    unique, so NeuG re-plans each one and the compilation is ~92% of the
    latency (see dialect.vector_literal). Normalizing a *query* vector is
    harmless: cosine distance is scale-invariant.
    """
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, datetime):
            out[key] = dialect.iso(value)
        elif value is None:
            out[key] = ''
        elif (
            isinstance(value, list)
            and value
            and all(dialect.is_float_scalar(x) for x in value)
        ):
            out[key] = dialect.l2_normalize([float(x) for x in value])
        else:
            out[key] = value
    return out


class NeuGDriverSession(GraphDriverSession):
    provider = GraphProvider.NEUG

    def __init__(self, driver: 'NeuGDriver'):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # NeuG has no server-side session state to release.
        pass

    async def close(self):
        # The underlying connection is owned by the driver and shared across
        # sessions; do not close it here.
        pass

    async def execute_write(self, func, *args, **kwargs):
        # NeuG has no multi-statement transactions; run the callback directly.
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            for cypher, params in query:
                await self.driver.execute_query(cypher, **params)
        else:
            await self.driver.execute_query(query, **kwargs)
        return None


class NeuGDriver(GraphDriver):
    """Embedded NeuG graph database driver.

    NeuG is an embedded graph database with a Cypher-flavored query language.
    The driver owns a single database handle and connection; all queries are
    serialized through an asyncio lock and run on a worker thread, since the
    binding is synchronous and a single connection must not see concurrent use.
    """

    provider: GraphProvider = GraphProvider.NEUG

    def __init__(
        self,
        db_path: str = 'graphiti.neug',
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self._database = db_path
        self.embedding_dim = embedding_dim

        self._db = neug.Database(db_path, mode='w')
        self._conn = self._db.connect()
        self._lock = asyncio.Lock()

        # Tables + extensions + indexes must exist before any query runs.
        self._bootstrap_sync()

        self._entity_node_ops = NeuGEntityNodeOperations()
        self._episode_node_ops = NeuGEpisodeNodeOperations()
        self._community_node_ops = NeuGCommunityNodeOperations()
        self._saga_node_ops = NeuGSagaNodeOperations()
        self._entity_edge_ops = NeuGEntityEdgeOperations()
        self._episodic_edge_ops = NeuGEpisodicEdgeOperations()
        self._community_edge_ops = NeuGCommunityEdgeOperations()
        self._has_episode_edge_ops = NeuGHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = NeuGNextEpisodeEdgeOperations()
        self._search_ops = NeuGSearchOperations()
        self._graph_ops = NeuGGraphMaintenanceOperations()

    # --- internal synchronous helpers (run under self._lock or in __init__) ---

    def _exec_sync(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        exec_mode = dialect.access_mode_for(query)
        if params is not None:
            result = self._conn.execute(query, exec_mode, params)
        else:
            result = self._conn.execute(query, exec_mode)
        return _records_of(result)

    def _bootstrap_sync(self):
        """Create extensions, node/rel tables, and search indexes (idempotent)."""
        for ddl in schema.load_extension_ddl():
            self._exec_sync(ddl)
        for ddl in schema.node_table_ddl(self.embedding_dim):
            self._exec_sync(ddl)
        for ddl in schema.rel_table_ddl(self.embedding_dim):
            self._exec_sync(ddl)
        for ddl in schema.index_ddl(
            ENTITY_INDEX_NAME, EPISODE_INDEX_NAME, COMMUNITY_INDEX_NAME, ENTITY_EDGE_INDEX_NAME
        ):
            self._exec_sync(ddl)

    # --- GraphDriver interface ---

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        # NOTE: keep None-valued params — the binding accepts them (storing an
        # empty value), and dropping them would leave unbound $param references
        # in the query, which can crash the engine.
        params = {k: v for k, v in kwargs.items()}
        # These are Neo4j-specific routing knobs; NeuG ignores them.
        params.pop('database_', None)
        params.pop('routing_', None)
        params = _sanitize_params(params)

        async with self._lock:
            try:
                rows = await asyncio.to_thread(
                    self._exec_sync, cypher_query_, params if params else None
                )
            except Exception as e:
                trimmed = {k: (v[:5] if isinstance(v, list) else v) for k, v in params.items()}
                logger.error(f'Error executing NeuG query: {e}\n{cypher_query_}\n{trimmed}')
                raise

        return rows, None, None

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return NeuGDriverSession(self)

    async def close(self):
        # Close the connection first, but always close the DB even if that
        # throws. The DB close writes the shutdown checkpoint; upstream #962
        # (checkpoint-on-close failing with a rename error once an HNSW index
        # holds data) is fixed in NeuG 0.2.0 — verified: closing a populated
        # HNSW graph raises nothing, the manifest is written, and a reopen
        # reads the data back — so the old RuntimeError swallow is gone and a
        # genuine close failure now propagates.
        try:
            self._conn.close()
        finally:
            self._db.close()

    async def delete_all_indexes(self):
        for ddl in schema.index_drop_ddl(
            ENTITY_INDEX_NAME, EPISODE_INDEX_NAME, COMMUNITY_INDEX_NAME, ENTITY_EDGE_INDEX_NAME
        ):
            async with self._lock:
                await asyncio.to_thread(self._exec_sync, ddl)

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()
        else:
            # Retire the pre-multi-column FTS indexes even on a plain
            # (non-destructive) call: they are dead weight on any database
            # that still carries them, and this is the only migration hook an
            # existing database gets.
            for ddl in [
                f'DROP INDEX {name} IF EXISTS;'
                for name in schema.legacy_index_names(ENTITY_INDEX_NAME, EPISODE_INDEX_NAME)
            ]:
                async with self._lock:
                    await asyncio.to_thread(self._exec_sync, ddl)
        for ddl in schema.index_ddl(
            ENTITY_INDEX_NAME, EPISODE_INDEX_NAME, COMMUNITY_INDEX_NAME, ENTITY_EDGE_INDEX_NAME
        ):
            async with self._lock:
                await asyncio.to_thread(self._exec_sync, ddl)

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops
