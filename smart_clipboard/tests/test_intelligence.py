"""Tests for LiteLLM intelligence layer."""

import pytest
from unittest.mock import patch, Mock, AsyncMock

from smart_clipboard.core.intelligence import UniversalIntelligence


class TestClipIntelligence:
    """Test AI intelligence functionality."""

    @pytest.fixture
    def intelligence(self):
        """Create intelligence instance for testing."""
        return UniversalIntelligence()

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, intelligence):
        """Test successful embedding generation."""
        mock_response = Mock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch("core.intelligence.aembedding", return_value=mock_response):
            result = await intelligence.generate_embedding("test content")

            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embedding_fallback(self, intelligence):
        """Test embedding fallback when LiteLLM fails."""
        with patch("core.intelligence.aembedding", side_effect=Exception("API Error")):
            result = await intelligence.generate_embedding("test content")

            # Should return fallback embedding
            assert result is not None
            assert len(result) == 1536  # OpenAI embedding dimension
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_generate_embedding_without_litellm(self):
        """Test embedding generation when LiteLLM is not available."""
        with patch("core.intelligence.LITELLM_AVAILABLE", False):
            intelligence = ClipIntelligence()
            result = await intelligence.generate_embedding("test content")

            # Should use fallback implementation
            assert result is not None
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_generate_tags_success(self, intelligence):
        """Test successful tag generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '["python", "web", "tutorial"]'

        with patch("core.intelligence.acompletion", return_value=mock_response):
            result = await intelligence.generate_tags("Python web development tutorial")

            assert result == ["python", "web", "tutorial"]

    @pytest.mark.asyncio
    async def test_generate_tags_malformed_json(self, intelligence):
        """Test tag generation with malformed JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = 'Tags: "python", "web", "tutorial"'

        with patch("core.intelligence.acompletion", return_value=mock_response):
            result = await intelligence.generate_tags("Python web development tutorial")

            # Should extract tags from malformed response
            assert "python" in result
            assert "web" in result
            assert "tutorial" in result

    @pytest.mark.asyncio
    async def test_generate_tags_fallback(self, intelligence):
        """Test tag generation fallback."""
        with patch("core.intelligence.acompletion", side_effect=Exception("API Error")):
            result = await intelligence.generate_tags(
                "def hello_world(): print('Hello, World!')"
            )

            # Should use rule-based fallback
            assert "python" in result  # Should detect Python code

    @pytest.mark.asyncio
    async def test_should_store_valuable_content(self, intelligence):
        """Test content value assessment for agent mode."""
        # Code content - should store
        code_content = """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        """
        assert await intelligence.should_store(code_content) is True

        # Tutorial content - should store
        tutorial_content = """
        How to set up Docker for Python development:
        1. Install Docker Desktop
        2. Create a Dockerfile
        3. Build the image
        Example: docker build -t my-app .
        """
        assert await intelligence.should_store(tutorial_content) is True

        # Documentation - should store
        doc_content = """
        API Documentation for user authentication:
        POST /auth/login
        Parameters: username, password
        Returns: JWT token for authorization
        """
        assert await intelligence.should_store(doc_content) is True

    @pytest.mark.asyncio
    async def test_should_store_uninteresting_content(self, intelligence):
        """Test filtering of uninteresting content."""
        # Too short
        assert await intelligence.should_store("short") is False

        # Too long
        long_content = "x" * 25000
        assert await intelligence.should_store(long_content) is False

        # Common uninteresting patterns
        assert await intelligence.should_store("Loading... please wait") is False
        assert await intelligence.should_store("Error 404 - Page not found") is False
        assert await intelligence.should_store("Click here to subscribe now!") is False

        # Generic text without value indicators
        generic_content = "This is just some regular text without any particular value or interesting information."
        assert await intelligence.should_store(generic_content) is False

    def test_fallback_tags_programming_languages(self, intelligence):
        """Test rule-based tag generation for programming languages."""
        # Python
        python_code = "def hello(): import os; print('Hello')"
        tags = intelligence._fallback_tags(python_code)
        assert "python" in tags

        # JavaScript
        js_code = "function hello() { const message = 'Hello'; console.log(message); }"
        tags = intelligence._fallback_tags(js_code)
        assert "javascript" in tags

        # TypeScript
        ts_code = "interface User { name: string; age: number; }"
        tags = intelligence._fallback_tags(ts_code)
        assert "typescript" in tags

        # Docker
        docker_content = (
            "FROM python:3.9\nRUN pip install requirements.txt\nCMD python app.py"
        )
        tags = intelligence._fallback_tags(docker_content)
        assert "docker" in tags

    def test_fallback_tags_technologies(self, intelligence):
        """Test rule-based tag generation for technologies."""
        # Git
        git_content = "git commit -m 'Initial commit'\ngit push origin main"
        tags = intelligence._fallback_tags(git_content)
        assert "git" in tags

        # Database
        sql_content = "SELECT * FROM users WHERE age > 18 ORDER BY name"
        tags = intelligence._fallback_tags(sql_content)
        assert "database" in tags

        # Web
        web_content = "GET /api/users HTTP/1.1\nContent-Type: application/json"
        tags = intelligence._fallback_tags(web_content)
        assert "web" in tags

        # Cloud
        cloud_content = "aws s3 cp file.txt s3://my-bucket/"
        tags = intelligence._fallback_tags(cloud_content)
        assert "cloud" in tags

    def test_fallback_tags_content_types(self, intelligence):
        """Test rule-based tag generation for content types."""
        # Tutorial
        tutorial_content = "How to install Python: Step 1, download Python..."
        tags = intelligence._fallback_tags(tutorial_content)
        assert "tutorial" in tags

        # Troubleshooting
        error_content = (
            "Error: Module not found. Fix: Install the missing module using pip"
        )
        tags = intelligence._fallback_tags(error_content)
        assert "troubleshooting" in tags

        # Configuration
        config_content = (
            "Setup your development environment: Configure your IDE settings"
        )
        tags = intelligence._fallback_tags(config_content)
        assert "configuration" in tags

    def test_extract_tags_from_text(self, intelligence):
        """Test tag extraction from malformed responses."""
        # Quoted tags
        quoted_text = 'Here are the tags: "python", "web", "development"'
        tags = intelligence._extract_tags_from_text(quoted_text)
        assert "python" in tags
        assert "web" in tags
        assert "development" in tags

        # Comma-separated tags
        comma_text = "Tags: python, web development, tutorial, api"
        tags = intelligence._extract_tags_from_text(comma_text)
        assert len(tags) > 0

        # No valid tags found
        no_tags_text = "This text has no extractable tags in it"
        tags = intelligence._extract_tags_from_text(no_tags_text)
        assert "untagged" in tags

    def test_fallback_embedding_consistency(self, intelligence):
        """Test that fallback embeddings are consistent."""
        content = "Test content for embedding"

        # Same content should produce same embedding
        embedding1 = intelligence._fallback_embedding(content)
        embedding2 = intelligence._fallback_embedding(content)

        assert embedding1 == embedding2
        assert len(embedding1) == 1536
        assert all(-1 <= x <= 1 for x in embedding1)  # Values in expected range

        # Different content should produce different embeddings
        different_content = "Different test content"
        embedding3 = intelligence._fallback_embedding(different_content)

        assert embedding1 != embedding3

    @pytest.mark.asyncio
    async def test_long_content_truncation(self, intelligence):
        """Test that long content is properly truncated."""
        # Very long content
        long_content = "x" * 15000

        with patch("core.intelligence.aembedding") as mock_embedding:
            mock_response = Mock()
            mock_response.data = [{"embedding": [0.1] * 1536}]
            mock_embedding.return_value = mock_response

            await intelligence.generate_embedding(long_content)

            # Should be called with truncated content
            call_args = mock_embedding.call_args[1]["input"]
            assert len(call_args) <= 8000

    @pytest.mark.asyncio
    async def test_tag_generation_limit(self, intelligence):
        """Test that tag generation respects limits."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        # Return more than 5 tags
        mock_response.choices[0].message.content = (
            '["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"]'
        )

        with patch("core.intelligence.acompletion", return_value=mock_response):
            result = await intelligence.generate_tags("test content")

            # Should limit to 5 tags
            assert len(result) <= 5
