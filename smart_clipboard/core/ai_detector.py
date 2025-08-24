"""AI Provider Detection and Configuration Utilities."""

import os
import json
import asyncio
import httpx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AIProvider:
    """Represents an AI provider configuration."""

    name: str
    type: str  # "local" or "cloud"
    api_base: str
    api_key: Optional[str] = None
    models: List[str] = None
    embedding_models: List[str] = None
    chat_models: List[str] = None
    available: bool = False
    latency_ms: Optional[float] = None


class AIDetector:
    """Detects and configures available AI providers."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.detected_providers = []

    async def detect_all_providers(self) -> List[AIProvider]:
        """Detect all available AI providers (local and cloud)."""
        providers = []

        # Detect local providers
        local_providers = await self._detect_local_providers()
        providers.extend(local_providers)

        # Detect cloud providers
        cloud_providers = await self._detect_cloud_providers()
        providers.extend(cloud_providers)

        self.detected_providers = providers
        return providers

    async def _detect_local_providers(self) -> List[AIProvider]:
        """Detect local AI servers."""
        local_providers = []

        # Common local AI server configurations
        local_configs = [
            {
                "name": "lmstudio",
                "api_base": os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234"),
                "models_endpoint": "/v1/models",
            },
            {
                "name": "lmstudio_network",
                "api_base": "http://192.168.86.20:1234",  # User's detected server
                "models_endpoint": "/v1/models",
            },
            {
                "name": "ollama",
                "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                "models_endpoint": "/api/tags",
            },
            {
                "name": "vllm",
                "api_base": os.getenv("VLLM_API_BASE", "http://localhost:8000"),
                "models_endpoint": "/v1/models",
            },
            {
                "name": "textgen_webui",
                "api_base": os.getenv("TEXTGEN_API_BASE", "http://localhost:5000"),
                "models_endpoint": "/v1/models",
            },
            {
                "name": "koboldcpp",
                "api_base": "http://localhost:5001",
                "models_endpoint": "/v1/models",
            },
        ]

        # Test each local provider
        tasks = []
        for config in local_configs:
            task = self._test_local_provider(config)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, AIProvider) and result.available:
                local_providers.append(result)

        return local_providers

    async def _test_local_provider(
        self, config: Dict[str, str]
    ) -> Optional[AIProvider]:
        """Test if a local provider is available."""
        try:
            api_base = config["api_base"]
            models_endpoint = config["models_endpoint"]

            start_time = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{api_base}{models_endpoint}")

                if response.status_code == 200:
                    end_time = asyncio.get_event_loop().time()
                    latency = (end_time - start_time) * 1000

                    models_data = response.json()
                    models, embedding_models, chat_models = self._parse_models(
                        models_data, config["name"]
                    )

                    return AIProvider(
                        name=config["name"],
                        type="local",
                        api_base=api_base,
                        api_key="",  # Local providers typically don't need keys
                        models=models,
                        embedding_models=embedding_models,
                        chat_models=chat_models,
                        available=True,
                        latency_ms=latency,
                    )
        except Exception as e:
            logger.debug(
                f"Provider {config['name']} at {config['api_base']} not available: {e}"
            )

        return AIProvider(
            name=config["name"],
            type="local",
            api_base=config["api_base"],
            available=False,
        )

    def _parse_models(
        self, models_data: Dict, provider_name: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parse model data and categorize models."""
        all_models = []
        embedding_models = []
        chat_models = []

        if provider_name == "ollama":
            # Ollama format
            models_list = models_data.get("models", [])
            for model in models_list:
                model_name = model.get("name", "")
                all_models.append(model_name)

                # Categorize by name patterns
                if "embed" in model_name.lower():
                    embedding_models.append(model_name)
                else:
                    chat_models.append(model_name)
        else:
            # OpenAI-compatible format (LM Studio, vLLM, etc.)
            models_list = models_data.get("data", [])
            for model in models_list:
                model_id = model.get("id", "")
                all_models.append(model_id)

                # Categorize by name patterns
                if any(
                    embed_pattern in model_id.lower()
                    for embed_pattern in ["embed", "embedding", "text-embedding"]
                ):
                    embedding_models.append(model_id)
                else:
                    chat_models.append(model_id)

        return all_models, embedding_models, chat_models

    async def _detect_cloud_providers(self) -> List[AIProvider]:
        """Detect available cloud providers based on environment variables."""
        cloud_providers = []

        # Cloud provider configurations
        cloud_configs = [
            {
                "name": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "api_base": "https://api.openai.com/v1",
                "test_endpoint": "/models",
                "embedding_models": [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ],
                "chat_models": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                ],
            },
            {
                "name": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "api_base": "https://api.anthropic.com",
                "test_endpoint": "/v1/messages",
                "embedding_models": [],  # Anthropic doesn't provide embeddings
                "chat_models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-haiku-20240307",
                ],
            },
            {
                "name": "google",
                "api_key_env": "GOOGLE_API_KEY",
                "api_base": "https://generativelanguage.googleapis.com/v1",
                "test_endpoint": "/models",
                "embedding_models": ["models/embedding-001"],
                "chat_models": [
                    "models/gemini-1.5-pro",
                    "models/gemini-1.5-flash",
                    "models/gemini-pro",
                ],
            },
            {
                "name": "cohere",
                "api_key_env": "COHERE_API_KEY",
                "api_base": "https://api.cohere.ai",
                "test_endpoint": "/v1/models",
                "embedding_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
                "chat_models": ["command-r-plus", "command-r", "command"],
            },
            {
                "name": "groq",
                "api_key_env": "GROQ_API_KEY",
                "api_base": "https://api.groq.com/openai/v1",
                "test_endpoint": "/models",
                "embedding_models": [],
                "chat_models": [
                    "llama-3.1-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                ],
            },
        ]

        for config in cloud_configs:
            api_key = os.getenv(config["api_key_env"])
            if api_key:
                provider = await self._test_cloud_provider(config, api_key)
                if provider.available:
                    cloud_providers.append(provider)

        return cloud_providers

    async def _test_cloud_provider(self, config: Dict, api_key: str) -> AIProvider:
        """Test if a cloud provider is available."""
        try:
            headers = self._get_auth_headers(config["name"], api_key)

            start_time = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Simple test call
                if config["name"] == "anthropic":
                    # Anthropic has different API structure, just mark as available if key exists
                    latency = 100  # Estimated
                else:
                    response = await client.get(
                        f"{config['api_base']}{config['test_endpoint']}",
                        headers=headers,
                    )

                    if response.status_code in [
                        200,
                        401,
                    ]:  # 401 means auth issue but API is reachable
                        end_time = asyncio.get_event_loop().time()
                        latency = (end_time - start_time) * 1000
                    else:
                        raise Exception(f"HTTP {response.status_code}")

                return AIProvider(
                    name=config["name"],
                    type="cloud",
                    api_base=config["api_base"],
                    api_key=api_key,
                    embedding_models=config["embedding_models"],
                    chat_models=config["chat_models"],
                    available=True,
                    latency_ms=latency,
                )

        except Exception as e:
            logger.debug(f"Cloud provider {config['name']} not available: {e}")

        return AIProvider(
            name=config["name"],
            type="cloud",
            api_base=config["api_base"],
            api_key=api_key,
            available=False,
        )

    def _get_auth_headers(self, provider: str, api_key: str) -> Dict[str, str]:
        """Get authentication headers for different providers."""
        if provider == "anthropic":
            return {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
        elif provider == "google":
            return {"Authorization": f"Bearer {api_key}"}
        elif provider == "cohere":
            return {"Authorization": f"Bearer {api_key}"}
        else:
            # Default OpenAI-compatible
            return {"Authorization": f"Bearer {api_key}"}

    def get_best_providers(self) -> Dict[str, AIProvider]:
        """Get the best available providers for different tasks."""
        best = {
            "embedding": None,
            "chat": None,
            "local_embedding": None,
            "local_chat": None,
            "cloud_embedding": None,
            "cloud_chat": None,
        }

        # Sort providers by preference (local first, then by latency)
        local_providers = [
            p for p in self.detected_providers if p.type == "local" and p.available
        ]
        cloud_providers = [
            p for p in self.detected_providers if p.type == "cloud" and p.available
        ]

        # Sort by latency
        local_providers.sort(key=lambda x: x.latency_ms or float("inf"))
        cloud_providers.sort(key=lambda x: x.latency_ms or float("inf"))

        # Find best embedding providers
        for provider in local_providers:
            if provider.embedding_models and not best["local_embedding"]:
                best["local_embedding"] = provider

        for provider in cloud_providers:
            if provider.embedding_models and not best["cloud_embedding"]:
                best["cloud_embedding"] = provider

        # Find best chat providers
        for provider in local_providers:
            if provider.chat_models and not best["local_chat"]:
                best["local_chat"] = provider

        for provider in cloud_providers:
            if provider.chat_models and not best["cloud_chat"]:
                best["cloud_chat"] = provider

        # Set overall best (prefer local)
        best["embedding"] = best["local_embedding"] or best["cloud_embedding"]
        best["chat"] = best["local_chat"] or best["cloud_chat"]

        return best

    def generate_env_config(self) -> str:
        """Generate .env configuration based on detected providers."""
        lines = [
            "# Smart Clipboard - Auto-detected Configuration",
            "# Generated by AI provider detection",
            "",
        ]

        best = self.get_best_providers()

        if best["embedding"]:
            lines.extend(
                [
                    "# Best detected embedding provider",
                    f"# Provider: {best['embedding'].name} ({best['embedding'].type})",
                    (
                        f"# Latency: {best['embedding'].latency_ms:.1f}ms"
                        if best["embedding"].latency_ms
                        else ""
                    ),
                    f"# Available models: {', '.join(best['embedding'].embedding_models[:3])}",
                ]
            )

        if best["chat"]:
            lines.extend(
                [
                    "",
                    "# Best detected chat provider",
                    f"# Provider: {best['chat'].name} ({best['chat'].type})",
                    (
                        f"# Latency: {best['chat'].latency_ms:.1f}ms"
                        if best["chat"].latency_ms
                        else ""
                    ),
                    f"# Available models: {', '.join(best['chat'].chat_models[:3])}",
                ]
            )

        # Add detected API bases for local providers
        local_providers = [
            p for p in self.detected_providers if p.type == "local" and p.available
        ]
        if local_providers:
            lines.extend(["", "# Detected local providers"])
            for provider in local_providers:
                lines.append(f"{provider.name.upper()}_API_BASE={provider.api_base}")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a human-readable summary of detected providers."""
        if not self.detected_providers:
            return "No AI providers detected. Run detect_all_providers() first."

        available = [p for p in self.detected_providers if p.available]
        local_available = [p for p in available if p.type == "local"]
        cloud_available = [p for p in available if p.type == "cloud"]

        lines = [f"🔍 Detected {len(available)} available AI providers:", ""]

        if local_available:
            lines.append("🏠 Local Providers:")
            for provider in local_available:
                latency_info = (
                    f" ({provider.latency_ms:.0f}ms)" if provider.latency_ms else ""
                )
                lines.append(
                    f"  ✅ {provider.name}{latency_info} - {len(provider.models or [])} models"
                )

        if cloud_available:
            lines.append("☁️  Cloud Providers:")
            for provider in cloud_available:
                latency_info = (
                    f" ({provider.latency_ms:.0f}ms)" if provider.latency_ms else ""
                )
                lines.append(f"  ✅ {provider.name}{latency_info} - API key configured")

        if not available:
            lines.append(
                "❌ No providers available. Configure API keys or start local AI servers."
            )

        return "\n".join(lines)
