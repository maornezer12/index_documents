#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL database operations.
"""

import logging
from typing import Sequence, Tuple

import psycopg2
from psycopg2.extensions import connection as PGConnection

def ensure_table_exists(conn: PGConnection, logger: logging.Logger, table_name: str) -> None:
    """Create the table and indexes if they do not exist.
    
    Args:
        conn: PostgreSQL connection
        logger: Logger instance
        table_name: Name of the table to ensure existence
    """
    with conn.cursor() as cur:
        # Create table
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              id BIGSERIAL PRIMARY KEY,
              filename TEXT NOT NULL,
              strategy_split TEXT NOT NULL,
              chunk_text TEXT NOT NULL,
              embedding DOUBLE PRECISION[] NOT NULL,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        )
        # Create indexes
        cur.execute(f"CREATE INDEX IF NOT EXISTS ix_filename ON {table_name}(filename);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS ix_strategy_split ON {table_name}(strategy_split);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS ix_created_at ON {table_name}(created_at);")
        logger.debug("Table and indexes verified/created")
    conn.commit()


def delete_existing_chunks(conn: PGConnection, filename: str, strategy: str, logger: logging.Logger, table_name: str) -> int:
    """Delete existing chunks for a filename and strategy combination to prevent duplicates.
    
    Only deletes chunks if both filename AND strategy match. This allows the same file
    to be indexed with different strategies simultaneously.
    
    Args:
        conn: PostgreSQL connection
        filename: Filename to delete chunks for
        strategy: Chunking strategy to delete chunks for
        logger: Logger instance
        table_name: Name of the table to delete from
    
    Returns:
        Number of deleted rows
    """
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table_name} WHERE filename = %s AND strategy_split = %s", (filename, strategy))
        deleted = cur.rowcount
        if deleted > 0:
            logger.info("Deleted %d existing chunks for filename '%s' with strategy '%s'", deleted, filename, strategy)
        else:
            logger.debug("No existing chunks found for filename '%s' with strategy '%s'", filename, strategy)
    conn.commit()
    return deleted


def insert_chunks(conn: PGConnection, rows: list[Tuple[str, str, str, Sequence[float]]], logger: logging.Logger, table_name: str) -> int:
    """Insert chunk rows into the database using executemany.

    Args:
        conn: PostgreSQL connection
        rows: List of tuples (filename, strategy_split, chunk_text, embedding)
        logger: Logger instance
        table_name: Name of the table to insert into

    Returns:
        Number of inserted rows
    """
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {table_name} (filename, strategy_split, chunk_text, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            rows,
        )
        logger.debug("Inserted %d chunks into database", len(rows))
    conn.commit()
    return len(rows)

