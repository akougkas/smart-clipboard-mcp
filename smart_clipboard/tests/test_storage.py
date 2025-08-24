"""Tests for LanceDB storage backend."""

import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, Mock

from smart_clipboard.core.storage import ClipStorage


class TestClipStorage:
    """Test LanceDB storage functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()

        # Mock the storage path
        with patch.object(ClipStorage, "__init__", lambda self: None):
            storage = ClipStorage()
            storage.db_path = temp_dir
            storage.db = None
            storage.table = None
            storage._initialized = False

        yield storage

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_storage_initialization(self, temp_storage):
        """Test storage initialization creates necessary structures."""
        # Mock LanceDB for initialization test
        mock_db = Mock()
        mock_table = Mock()

        with patch("lancedb.connect", return_value=mock_db):
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = mock_table
            mock_table.delete = Mock()

            await temp_storage._ensure_initialized()

            assert temp_storage._initialized is True
            mock_db.create_table.assert_called_once()
            mock_table.delete.assert_called_once_with("id = 'init'")

    @pytest.mark.asyncio
    async def test_store_clip(self, temp_storage):
        """Test storing a clip with embedding."""
        # Mock LanceDB components
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = (
            []
        )
        mock_table.add = Mock()

        clip_id = "test_clip_123"
        content = "Test content for storage"
        embedding = [0.1] * 1536
        tags = ["test", "storage"]
        metadata = {"source": "manual", "is_agent": False}

        result = await temp_storage.store(clip_id, content, embedding, tags, metadata)

        assert result is True
        mock_table.add.assert_called_once()

        # Verify the record structure
        call_args = mock_table.add.call_args[0][0][0]
        assert call_args["id"] == clip_id
        assert call_args["content"] == content
        assert call_args["embedding"] == embedding
        assert call_args["tags"] == tags

    @pytest.mark.asyncio
    async def test_store_clip_without_embedding(self, temp_storage):
        """Test storing clip without embedding uses fallback."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = (
            []
        )
        mock_table.add = Mock()

        result = await temp_storage.store("test_clip", "content", None, [], {})

        assert result is True
        call_args = mock_table.add.call_args[0][0][0]
        # Should use zero vector as fallback
        assert call_args["embedding"] == [0.0] * 1536

    @pytest.mark.asyncio
    async def test_search_clips(self, temp_storage):
        """Test semantic search functionality."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        # Mock search results
        mock_results = [
            {
                "id": "clip1",
                "content": "First test clip",
                "tags": ["test"],
                "_distance": 0.1,
                "source": "manual",
                "timestamp": 1234567890,
                "is_agent": False,
            },
            {
                "id": "clip2",
                "content": "Second test clip",
                "tags": ["example"],
                "_distance": 0.3,
                "source": "agent",
                "timestamp": 1234567891,
                "is_agent": True,
            },
        ]

        mock_table.search.return_value.limit.return_value.to_list.return_value = (
            mock_results
        )

        query_embedding = [0.5] * 1536
        results = await temp_storage.search(query_embedding, limit=5)

        assert len(results) == 2

        # Verify result structure and similarity calculation
        assert results[0]["id"] == "clip1"
        assert results[0]["score"] == 0.9  # 1.0 - 0.1 distance
        assert results[1]["score"] == 0.7  # 1.0 - 0.3 distance

        # Verify metadata is included
        assert results[0]["metadata"]["source"] == "manual"
        assert results[1]["metadata"]["is_agent"] is True

    @pytest.mark.asyncio
    async def test_list_recent_clips(self, temp_storage):
        """Test listing recent clips with proper sorting."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        # Mock pandas DataFrame
        import pandas as pd

        mock_data = pd.DataFrame(
            [
                {
                    "id": "clip1",
                    "content": "First clip with longer content for truncation testing",
                    "tags": ["test"],
                    "timestamp": 1234567890,
                    "source": "manual",
                },
                {
                    "id": "clip2",
                    "content": "Second clip",
                    "tags": ["recent"],
                    "timestamp": 1234567900,  # More recent
                    "source": "agent",
                },
            ]
        )

        mock_table.to_pandas.return_value = mock_data

        results = await temp_storage.list_recent(limit=2)

        assert len(results) == 2
        # Should be sorted by timestamp (most recent first)
        assert results[0]["id"] == "clip2"
        assert results[1]["id"] == "clip1"

        # Content should be truncated if too long
        assert len(results[1]["content"]) <= 203  # 200 + "..."

    @pytest.mark.asyncio
    async def test_remove_clip(self, temp_storage):
        """Test clip removal functionality."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        clip_id = "clip_to_remove"

        # Test successful removal
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
            {"id": clip_id}
        ]
        mock_table.delete = Mock()

        result = await temp_storage.remove(clip_id)

        assert result is True
        mock_table.delete.assert_called_once_with(f"id = '{clip_id}'")

        # Test clip not found
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = (
            []
        )

        result = await temp_storage.remove("nonexistent_clip")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_storage):
        """Test statistics generation."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        # Mock pandas DataFrame with test data
        import pandas as pd

        mock_data = pd.DataFrame(
            [
                {
                    "id": "clip1",
                    "content": "Short",
                    "tags": ["python", "test"],
                    "is_agent": True,
                    "timestamp": 1234567890,
                },
                {
                    "id": "clip2",
                    "content": "Much longer content for testing",
                    "tags": ["javascript", "test"],
                    "is_agent": False,
                    "timestamp": 1234567900,
                },
                {
                    "id": "clip3",
                    "content": "Medium length content",
                    "tags": ["python"],
                    "is_agent": True,
                    "timestamp": 1234567880,
                },
            ]
        )

        mock_table.to_pandas.return_value = mock_data

        stats = await temp_storage.get_stats()

        assert stats["total_clips"] == 3
        assert stats["agent_clips"] == 2
        assert stats["manual_clips"] == 1
        assert stats["avg_content_length"] > 0
        assert len(stats["top_tags"]) > 0
        assert stats["storage_path"] == temp_storage.db_path

        # Check that python is the most common tag
        top_tag = stats["top_tags"][0]
        assert top_tag[0] == "python"
        assert top_tag[1] == 2  # Appears in 2 clips

    @pytest.mark.asyncio
    async def test_get_clip(self, temp_storage):
        """Test retrieving a specific clip by ID."""
        mock_table = Mock()
        temp_storage.table = mock_table
        temp_storage._initialized = True

        clip_id = "specific_clip"
        mock_clip_data = [
            {
                "id": clip_id,
                "content": "Specific clip content",
                "tags": ["specific"],
                "timestamp": 1234567890,
                "source": "manual",
                "is_agent": False,
            }
        ]

        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = (
            mock_clip_data
        )

        result = await temp_storage.get_clip(clip_id)

        assert result is not None
        assert result["id"] == clip_id
        assert result["content"] == "Specific clip content"
        assert result["tags"] == ["specific"]

        # Test clip not found
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = (
            []
        )

        result = await temp_storage.get_clip("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_storage):
        """Test error handling in storage operations."""
        temp_storage._initialized = True
        temp_storage.table = None  # Force error

        # All operations should handle errors gracefully
        result = await temp_storage.store("id", "content", [], [], {})
        assert result is False

        results = await temp_storage.search([0.1] * 1536, 5)
        assert results == []

        results = await temp_storage.list_recent(10)
        assert results == []

        result = await temp_storage.remove("id")
        assert result is False

        stats = await temp_storage.get_stats()
        assert "error" in stats
