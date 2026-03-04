"""Database layer for the AI Video Intelligence Platform.

Provides PostgreSQL + pgvector operations for actor
embeddings storage, retrieval, and similarity search.
"""

import logging
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# --- constants ---

VECTOR_DIMENSIONS: int = 512
IVFFLAT_LISTS: int = 100


def init_db(connection_string: str) -> None:
    """Create tables and extensions if they do not exist.

    Installs the pgvector extension and creates the actors
    table with an IVFFlat cosine-similarity index.  Safe to
    call repeatedly — all statements are idempotent.

    Args:
        connection_string (str): A libpq connection string,
            e.g. ``postgresql://user:pass@host:port/db``.

    Raises:
        psycopg2.OperationalError: If the database cannot
            be reached.
        psycopg2.DatabaseError: If schema creation fails.
    """
    try:
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS actors (
                    id         SERIAL PRIMARY KEY,
                    name       TEXT NOT NULL UNIQUE,
                    embedding  vector(%s),
                    photo_path TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """,
                (VECTOR_DIMENSIONS,),
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS actors_embedding_idx
                ON actors
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = %s);
                """,
                (IVFFLAT_LISTS,),
            )
        conn.close()
        logger.info("Database initialised successfully.")
    except psycopg2.OperationalError as exc:
        raise psycopg2.OperationalError(
            "Could not connect to PostgreSQL during init_db."
        ) from exc
    except psycopg2.DatabaseError as exc:
        raise psycopg2.DatabaseError(
            "Schema creation failed in init_db."
        ) from exc


def insert_actor(
    name: str,
    embedding: list[float],
    photo_path: str,
    conn: Any,
) -> int:
    """Insert or update an actor record with its embedding.

    Uses ``ON CONFLICT (name) DO UPDATE`` so repeated calls
    for the same actor name are safe.

    Args:
        name (str): Display name of the actor.
        embedding (list[float]): A 512-dimensional
            L2-normalised embedding vector.
        photo_path (str): Path to the reference photo on
            disk.
        conn (Any): An open ``psycopg2`` connection.

    Returns:
        int: The ``id`` of the inserted or updated row.

    Raises:
        psycopg2.DatabaseError: If the insert fails.
    """
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO actors (name, embedding, photo_path)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT (name) DO UPDATE
                    SET embedding  = EXCLUDED.embedding,
                        photo_path = EXCLUDED.photo_path
                RETURNING id;
                """,
                (name, vec_str, photo_path),
            )
            row = cur.fetchone()
            conn.commit()
            if row is None:
                raise psycopg2.DatabaseError(
                    "INSERT returned no id for actor."
                )
            actor_id: int = row[0]
            return actor_id
    except psycopg2.DatabaseError as exc:
        conn.rollback()
        raise psycopg2.DatabaseError(
            f"Failed to insert actor '{name}'."
        ) from exc


def get_all_actors(conn: Any) -> list[dict[str, Any]]:
    """Retrieve all actor rows from the database.

    Args:
        conn (Any): An open ``psycopg2`` connection.

    Returns:
        list[dict[str, Any]]: Each dict contains ``id``,
            ``name``, and ``embedding`` (as a list of
            floats).

    Raises:
        psycopg2.DatabaseError: If the query fails.
    """
    try:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(
                "SELECT id, name, embedding FROM actors;"
            )
            rows = cur.fetchall()
        actors: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            # pgvector returns a string like '[0.1,0.2,...]';
            # parse it to a plain Python list of floats.
            raw_emb = row_dict.get("embedding")
            if isinstance(raw_emb, str):
                cleaned = raw_emb.strip("[]")
                row_dict["embedding"] = [
                    float(v) for v in cleaned.split(",") if v
                ]
            actors.append(row_dict)
        return actors
    except psycopg2.DatabaseError as exc:
        raise psycopg2.DatabaseError(
            "Failed to retrieve actors from database."
        ) from exc


def find_similar_actor(
    embedding: list[float],
    threshold: float,
    conn: Any,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Find actors whose embedding is cosine-similar enough.

    Uses the pgvector ``<=>`` (cosine distance) operator.
    Similarity = 1 - cosine_distance.

    Args:
        embedding (list[float]): Query embedding vector
            (512-d, L2-normalised).
        threshold (float): Minimum cosine similarity score
            to include a result (e.g. 0.45).
        conn (Any): An open ``psycopg2`` connection.
        limit (int): Maximum number of results to return.
            Defaults to 1.

    Returns:
        list[dict[str, Any]]: Each dict contains ``name``
            and ``score`` (float, higher is more similar).

    Raises:
        psycopg2.DatabaseError: If the query fails.
    """
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    try:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT
                    name,
                    1 - (embedding <=> %s::vector) AS score
                FROM actors
                WHERE
                    1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vec_str, vec_str, threshold, vec_str, limit),
            )
            rows = cur.fetchall()
        return [dict(row) for row in rows]
    except psycopg2.DatabaseError as exc:
        raise psycopg2.DatabaseError(
            "Vector similarity search failed."
        ) from exc
