#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for database utilities.
"""

from unittest.mock import Mock, MagicMock
import pytest

from utils import db_utils


@pytest.fixture
def mock_connection():
    """Create a mock PostgreSQL connection."""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = MagicMock()
    conn.cursor.return_value.__exit__.return_value = None
    return conn


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock()


class TestEnsureTableExists:
    """Tests for ensure_table_exists function."""

    def test_creates_table_if_not_exists(self, mock_connection, mock_logger):
        """Test that table is created if it doesn't exist."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        db_utils.ensure_table_exists(mock_connection, mock_logger, "test_table")
        
        # Verify CREATE TABLE was called
        calls = [call[0][0] for call in cursor_mock.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS test_table" in call for call in calls)
        mock_connection.commit.assert_called_once()

    def test_creates_indexes(self, mock_connection, mock_logger):
        """Test that indexes are created."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        db_utils.ensure_table_exists(mock_connection, mock_logger, "test_table")
        
        # Verify indexes were created
        calls = [call[0][0] for call in cursor_mock.execute.call_args_list]
        index_calls = [call for call in calls if "CREATE INDEX" in call]
        assert len(index_calls) >= 3  # Should create at least 3 indexes


class TestDeleteExistingChunks:
    """Tests for delete_existing_chunks function."""

    def test_deletes_chunks_for_filename_and_strategy(self, mock_connection, mock_logger):
        """Test that chunks for a filename and strategy combination are deleted."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        cursor_mock.rowcount = 5
        
        deleted = db_utils.delete_existing_chunks(mock_connection, "test.pdf", "fixed", mock_logger, "test_table")
        
        assert deleted == 5
        cursor_mock.execute.assert_called_once()
        call_args = cursor_mock.execute.call_args[0]
        assert "DELETE FROM test_table" in call_args[0]
        assert "filename = %s AND strategy_split = %s" in call_args[0]
        assert call_args[1] == ("test.pdf", "fixed")
        mock_connection.commit.assert_called_once()

    def test_handles_no_existing_chunks(self, mock_connection, mock_logger):
        """Test that function handles case when no chunks exist."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        cursor_mock.rowcount = 0
        
        deleted = db_utils.delete_existing_chunks(mock_connection, "test.pdf", "sentence", mock_logger, "test_table")
        
        assert deleted == 0
        mock_connection.commit.assert_called_once()

    def test_deletes_only_matching_strategy(self, mock_connection, mock_logger):
        """Test that only chunks with matching filename and strategy are deleted."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        cursor_mock.rowcount = 3
        
        deleted = db_utils.delete_existing_chunks(mock_connection, "test.pdf", "fixed", mock_logger, "test_table")
        
        assert deleted == 3
        # Verify SQL includes both filename and strategy
        call_args = cursor_mock.execute.call_args[0]
        assert "AND strategy_split = %s" in call_args[0]


class TestInsertChunks:
    """Tests for insert_chunks function."""

    def test_inserts_chunks(self, mock_connection, mock_logger):
        """Test that chunks are inserted correctly."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        rows = [
            ("file1.pdf", "fixed", "chunk1", [0.1, 0.2, 0.3]),
            ("file1.pdf", "fixed", "chunk2", [0.4, 0.5, 0.6]),
        ]
        
        inserted = db_utils.insert_chunks(mock_connection, rows, mock_logger, "test_table")
        
        assert inserted == 2
        cursor_mock.executemany.assert_called_once()
        call_args = cursor_mock.executemany.call_args
        assert "INSERT INTO test_table" in call_args[0][0]
        assert call_args[0][1] == rows
        mock_connection.commit.assert_called_once()

    def test_empty_rows_returns_zero(self, mock_connection, mock_logger):
        """Test that empty rows list returns 0."""
        inserted = db_utils.insert_chunks(mock_connection, [], mock_logger, "test_table")
        
        assert inserted == 0
        # Should not call executemany on empty list
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        cursor_mock.executemany.assert_not_called()

    def test_inserts_embedding_vectors(self, mock_connection, mock_logger):
        """Test that embedding vectors are inserted correctly."""
        cursor_mock = mock_connection.cursor.return_value.__enter__.return_value
        rows = [
            ("file.pdf", "sentence", "text", [0.1] * 768),  # 768-dim embedding
        ]
        
        db_utils.insert_chunks(mock_connection, rows, mock_logger, "test_table")
        
        # Verify the embedding array is in the row
        call_args = cursor_mock.executemany.call_args
        assert call_args[0][1][0][3] == [0.1] * 768
