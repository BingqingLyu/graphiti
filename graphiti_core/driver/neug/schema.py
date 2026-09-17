"""
Schema DDL for the NeuG backend.

Layout (validated end-to-end in the M0 spike, see proposal-neug-driver/m0/):

Node tables
    Entity / Episodic / Community / Saga  — the four Graphiti node kinds
    EdgeDoc — mirror node table carrying edge `fact` + `fact_embedding`.
              NeuG's FTS/HNSW indexes can only be created on node tables,
              so edge-level retrieval runs against this mirror and joins
              back to the real RELATES_TO edge by shared `uuid`. Maintained
              by the driver inside the same database (no external system).

Rel tables
    RELATES_TO     Entity -> Entity      (entity edges, full property set)
    MENTIONS       Episodic -> Entity
    HAS_MEMBER     Community -> Entity / Community -> Community
    HAS_EPISODE    Saga -> Episodic
    NEXT_EPISODE   Episodic -> Episodic

Indexes
    FTS   multi-property on Entity(name, summary) and Episodic(content,
          source, source_description) — one weighted bm25() ranks all columns
          in a single query; single-property on Community(name), EdgeDoc(fact)
    HNSW  on Entity.name_embedding, Community.name_embedding,
          EdgeDoc.fact_embedding (cosine)

Temporal fields are stored as STRING columns (fixed-width ISO-8601 UTC) so
that lexicographic comparison preserves ordering — see dialect.py. List-valued
fields (labels, entity_edges, episodes) use native STRING[] columns; dict-valued
fields (attributes, episode_metadata) are JSON-serialized into string columns.

Free-text node columns use VARCHAR(TEXT_COLUMN_MAX_LENGTH) rather than bare
STRING: STRING maps to VARCHAR(256) and every write path silently truncates
values longer than the column's max length (episode content and entity
summaries routinely exceed 256 bytes). 65535 is the engine's max varchar
length (uint16).

Edge tables (RELATES_TO, EdgeDoc) use VARCHAR(TEXT_COLUMN_MAX_LENGTH) for
their free-text columns (name, fact, attributes) too. NeuG varchars are
variable-length on disk — measured identical size for bare STRING vs
VARCHAR(65535) (ratio 1.00x), with no per-row max_length reservation — so
widening costs nothing and avoids the 256-byte silent truncation of facts.
"""

from __future__ import annotations

#: Engine max varchar length; bare STRING silently truncates at 256 bytes.
TEXT_COLUMN_MAX_LENGTH = 65535

#: Long-form node text column type (episode content, summaries, JSON blobs).
TEXT = f'VARCHAR({TEXT_COLUMN_MAX_LENGTH})'


def node_table_ddl(embedding_dim: int) -> list[str]:
    return [
        f"""
        CREATE NODE TABLE IF NOT EXISTS Entity (
            uuid STRING,
            name STRING,
            name_embedding FLOAT[{embedding_dim}],
            group_id STRING,
            summary {TEXT},
            labels STRING[],
            attributes {TEXT},
            created_at STRING,
            PRIMARY KEY (uuid)
        )
        """,
        f"""
        CREATE NODE TABLE IF NOT EXISTS Episodic (
            uuid STRING,
            name STRING,
            group_id STRING,
            source STRING,
            source_description {TEXT},
            content {TEXT},
            entity_edges STRING[],
            episode_metadata {TEXT},
            created_at STRING,
            valid_at STRING,
            PRIMARY KEY (uuid)
        )
        """,
        f"""
        CREATE NODE TABLE IF NOT EXISTS Community (
            uuid STRING,
            name STRING,
            name_embedding FLOAT[{embedding_dim}],
            group_id STRING,
            summary {TEXT},
            created_at STRING,
            PRIMARY KEY (uuid)
        )
        """,
        f"""
        CREATE NODE TABLE IF NOT EXISTS Saga (
            uuid STRING,
            name STRING,
            group_id STRING,
            summary {TEXT},
            first_episode_uuid STRING,
            last_episode_uuid STRING,
            last_summarized_at STRING,
            last_summarized_episode_valid_at STRING,
            created_at STRING,
            PRIMARY KEY (uuid)
        )
        """,
        f"""
        CREATE NODE TABLE IF NOT EXISTS EdgeDoc (
            uuid STRING,
            source_node_uuid STRING,
            target_node_uuid STRING,
            name {TEXT},
            fact {TEXT},
            fact_embedding FLOAT[{embedding_dim}],
            episodes STRING[],
            group_id STRING,
            attributes {TEXT},
            created_at STRING,
            expired_at STRING,
            valid_at STRING,
            invalid_at STRING,
            reference_time STRING,
            PRIMARY KEY (uuid)
        )
        """,
    ]


def rel_table_ddl(embedding_dim: int) -> list[str]:
    return [
        f"""
        CREATE REL TABLE IF NOT EXISTS RELATES_TO (
            FROM Entity TO Entity,
            uuid STRING,
            name {TEXT},
            fact {TEXT},
            fact_embedding FLOAT[{embedding_dim}],
            episodes STRING[],
            group_id STRING,
            attributes {TEXT},
            created_at STRING,
            expired_at STRING,
            valid_at STRING,
            invalid_at STRING,
            reference_time STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS MENTIONS (
            FROM Episodic TO Entity,
            uuid STRING,
            group_id STRING,
            created_at STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS HAS_MEMBER (
            FROM Community TO Entity,
            FROM Community TO Community,
            uuid STRING,
            group_id STRING,
            created_at STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS HAS_EPISODE (
            FROM Saga TO Episodic,
            uuid STRING,
            group_id STRING,
            created_at STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS NEXT_EPISODE (
            FROM Episodic TO Episodic,
            uuid STRING,
            group_id STRING,
            created_at STRING
        )
        """,
    ]


def load_extension_ddl() -> list[str]:
    return ['LOAD fts;', 'LOAD vector_search;']


def index_ddl(
    entity_index_name: str,
    episode_index_name: str,
    community_index_name: str,
    entity_edge_index_name: str,
) -> list[str]:
    """FTS + HNSW index creation statements.

    Names derive from Graphiti's standard index-name constants so
    multi-backend tooling keeps recognizable identifiers; suffixes keep each
    index unique within the database.

    Entity and Episodic use a single MULTI-PROPERTY FTS index (name+summary /
    content+source+source_description) so one weighted bm25() call ranks all
    columns in a single query, instead of one index per column ranked
    separately and merged client-side.

    The HNSW indexes are created with ``cosine_normalize = false``: the
    default (true) rewrites the vector column to store L2-normalized values,
    so read-back embeddings differ from what was written. The driver
    pre-normalizes every embedding before writing (dialect.l2_normalize),
    which satisfies the engine's precondition for the false setting and
    keeps stored values stable.
    """
    return [
        f"""
        CREATE INDEX {entity_index_name}_fts IF NOT EXISTS
        ON Entity USING FTS (name, summary)
        """,
        f"""
        CREATE INDEX {episode_index_name}_fts IF NOT EXISTS
        ON Episodic USING FTS (content, source, source_description)
        """,
        f"""
        CREATE INDEX {community_index_name}_name_fts IF NOT EXISTS
        ON Community USING FTS (name)
        """,
        f"""
        CREATE INDEX {entity_edge_index_name}_fact_fts IF NOT EXISTS
        ON EdgeDoc USING FTS (fact)
        """,
        f"""
        CREATE INDEX {entity_index_name}_name_hnsw IF NOT EXISTS
        ON Entity USING HNSW (name_embedding)
        WITH (metric = 'cosine', cosine_normalize = false)
        """,
        f"""
        CREATE INDEX {community_index_name}_name_hnsw IF NOT EXISTS
        ON Community USING HNSW (name_embedding)
        WITH (metric = 'cosine', cosine_normalize = false)
        """,
        f"""
        CREATE INDEX {entity_edge_index_name}_fact_hnsw IF NOT EXISTS
        ON EdgeDoc USING HNSW (fact_embedding)
        WITH (metric = 'cosine', cosine_normalize = false)
        """,
    ]


def index_drop_ddl(
    entity_index_name: str,
    episode_index_name: str,
    community_index_name: str,
    entity_edge_index_name: str,
) -> list[str]:
    names = [
        f'{entity_index_name}_fts',
        f'{episode_index_name}_fts',
        f'{community_index_name}_name_fts',
        f'{entity_edge_index_name}_fact_fts',
        f'{entity_index_name}_name_hnsw',
        f'{community_index_name}_name_hnsw',
        f'{entity_edge_index_name}_fact_hnsw',
    ]
    return [f'DROP INDEX {name} IF EXISTS;' for name in names + legacy_index_names(
        entity_index_name, episode_index_name
    )]


def legacy_index_names(entity_index_name: str, episode_index_name: str) -> list[str]:
    """Single-column FTS indexes superseded by the multi-column ones.

    ``entities_fts`` now covers (name, summary) and ``episodes_fts`` covers
    (content, source, source_description). A database created before that
    merge keeps the five old indexes *active* next to the new ones — nothing
    reads them, but every Entity write still maintains 3 FTS indexes and every
    Episodic write 4. Dropped unconditionally by build_indices_and_constraints
    so an existing database migrates on the next open; IF EXISTS makes it a
    no-op on databases that never had them.
    """
    return [
        f'{entity_index_name}_name_fts',
        f'{entity_index_name}_summary_fts',
        f'{episode_index_name}_content_fts',
        f'{episode_index_name}_source_fts',
        f'{episode_index_name}_source_description_fts',
    ]
