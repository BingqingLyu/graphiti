"""
Cypher-dialect helpers for the NeuG backend.

NeuG accepts a Cypher-flavored query language with a handful of differences
from Neo4j that this module papers over (all validated against the NeuG
v0.2.0-equivalent local build during the M0 spike):

- List parameters bind directly (`IN $list`); older engine builds
  segfaulted on them, so the driver requires a recent build.
- Unbound `$param` references can crash the engine, so every parameter a
  query mentions must be supplied (None is fine: the binding stores it as
  an empty value).
- `LIMIT` only accepts literals, never parameters.
- String literals escape quotes with backslash (`\'`); the SQL-style
  doubled quote (`''`) is a parse error.
- Unset STRING properties read back as `''` rather than NULL.
- `bm25()` scores are negative and ascending (smaller = more relevant).
- Datetimes are stored as fixed-width ISO-8601 UTC strings so that string
  comparison preserves temporal ordering.
- Statements are classified into NeuG access modes: DDL runs in `schema`,
  `LOAD` runs in `update`, read-only queries run in `read`, and everything
  else uses the default mode (which notably is the only mode where `MERGE`
  is accepted).
"""

from __future__ import annotations

import json
import math
import numbers
from datetime import datetime
from typing import Any


def iso(dt: datetime | None) -> str:
    """Serialize a datetime as a fixed-width, sortable ISO-8601 string.

    Fixed width (microseconds always present) matters: stored values are
    compared lexicographically in queries, and a mix of second-precision and
    microsecond-precision strings would sort incorrectly.

    The tzinfo is preserved verbatim (naive stays naive, aware keeps its
    offset) so that round-trips through the database compare equal to the
    original values, matching the other embedded drivers' behavior.
    """
    if dt is None:
        return ''
    return dt.isoformat(timespec='microseconds')


def parse_iso(value: str | None) -> datetime | None:
    """Parse a stored ISO-8601 string back into a datetime, preserving tzinfo."""
    if value is None or value == '':
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def esc(value: str) -> str:
    """Escape a string for inlining into a NeuG string literal.

    NeuG only accepts backslash escapes: the SQL-style doubled quote ('')
    is a parse error. Backslashes themselves must be doubled first.
    Retained for the rare literal that still gets inlined (FTS terms);
    list-valued filters bind parameters instead.
    """
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


def neug_in_filter(column: str, values: list[str], params: dict[str, Any]) -> str:
    """Render `<column> IN $param` and bind the values into ``params``.

    The parameter name derives from the column, suffixed on the unlikely
    chance of a clash with an existing key. `IN []` is rejected by the
    engine, so an empty list degrades to `false` (a match-nothing
    predicate that keeps caller-side empty-result semantics intact) and
    binds nothing.
    """
    if not values:
        return 'false'
    name = column.replace('.', '_')
    while name in params:
        name += '_2'
    params[name] = list(values)
    return f'{column} IN ${name}'


def neug_limit(limit: int | None) -> str:
    """Render a LIMIT clause; NeuG only accepts literals, never parameters."""
    if limit is None:
        return ''
    return f'LIMIT {int(limit)}'


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize an embedding before it is written.

    NeuG cosine HNSW indexes store L2-normalized vectors by default
    (``cosine_normalize = true`` rewrites the vector column at index
    creation). The driver pre-normalizes every embedding it writes and
    creates its indexes with ``cosine_normalize = false``, so stored values
    are stable regardless of index state. Cosine similarity is invariant
    under normalization, so retrieval results are unchanged.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def is_float_scalar(value: Any) -> bool:
    """True for real-valued scalars that can only be embedding components.

    Python ``float`` plus the numpy float scalars (``np.float32``/``float64``,
    which register as ``numbers.Real``). ``int`` and ``bool`` are excluded on
    purpose: int lists are never embeddings, they are `IN $list` parameters,
    and coercing them would silently change the query's semantics.
    """
    if isinstance(value, float):
        return True
    return isinstance(value, numbers.Real) and not isinstance(value, (int, bool))


def vector_literal(vector: list[float]) -> str:
    """Render an embedding as a FLOAT[N]-compatible literal.

    Every element must carry a decimal point: NeuG does not implicitly cast
    DOUBLE[N] literals with integer elements into FLOAT[N].

    Write path only (COPY / CREATE), where a literal is unavoidable. Search
    queries must bind ``$search_vector`` instead: an inlined literal makes
    every query's text unique, so NeuG re-parses and re-plans each call, and
    that compilation dominates the query (92.4% of latency measured on a
    20k-node 1024-dim HNSW table: 30.7ms for a first-seen literal, 2.35ms for
    a repeated one, 2.4-3.4ms bound; identical top-k, and the bound form still
    receives the ANN IndexScan rewrite).
    """
    parts = []
    for x in vector:
        s = repr(float(x))
        if '.' not in s and 'e' not in s and 'E' not in s and 'inf' not in s and 'nan' not in s:
            s += '.0'
        parts.append(s)
    return '[' + ', '.join(parts) + ']'


def json_col(value: Any) -> str:
    """Serialize a list/dict attribute into its JSON string column form."""
    if value is None:
        return ''
    return json.dumps(value, ensure_ascii=False)


def from_json_col(value: str | None, default: Any) -> Any:
    """Deserialize a JSON string column; unset columns come back as ''."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


_WRITE_MARKERS = ('CREATE', 'DELETE', 'DETACH', 'MERGE', 'SET ', 'COPY', 'INSERT')


def access_mode_for(query: str) -> str:
    """Map a statement onto NeuG's access modes ('' is the default mode).

    - DDL and SHOW_INDEXES need `schema`
    - `LOAD <extension>` needs `update`
    - pure reads use `read`
    - writes (incl. MERGE) use the default mode, which auto-classifies
      writes; the dedicated `update` mode is reserved for LOAD statements.
    """
    q = query.strip()
    upper = q.upper()
    if upper.startswith(
        (
            'CREATE NODE TABLE',
            'CREATE REL TABLE',
            'CREATE INDEX',
            'ALTER TABLE',
            'DROP TABLE',
            'DROP INDEX',
        )
    ):
        return 'schema'
    if upper.startswith('CALL SHOW_INDEXES'):
        return 'schema'
    if upper.startswith('LOAD '):
        return 'update'
    if any(marker in upper for marker in _WRITE_MARKERS):
        return ''
    return 'read'


def build_fts_query(query: str, max_query_length: int = 8000) -> str:
    """Turn raw search text into a safe NeuG FTS query string.

    Each whitespace-delimited token becomes a quoted phrase term; quoted
    terms keep punctuation from being parsed as FTS syntax. Terms are
    implicitly ANDed by NeuG. Returns '' when nothing searchable remains.
    """
    text = query.replace('\\', ' ')[:max_query_length]
    for ch in '"*(){}[]^~?:!|&':
        text = text.replace(ch, ' ')
    tokens = [t for t in text.split() if t.strip()]
    if not tokens:
        return ''
    return ' '.join('"' + t.replace('"', '') + '"' for t in tokens)
