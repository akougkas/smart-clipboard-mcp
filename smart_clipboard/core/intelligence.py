"""Universal AI Intelligence using LiteLLM Router for any provider."""

import json
import re
import os
import time
import logging
import hashlib
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from litellm import Router, aembedding, acompletion
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    Router = None
    LITELLM_AVAILABLE = False

from .ai_detector import AIDetector, AIProvider

logger = logging.getLogger(__name__)


class UniversalIntelligence:
    """Universal AI client with automatic provider detection and routing."""

    def __init__(self):
        self.router = None
        self.embedding_model_name = "universal-embedding"
        self.chat_model_name = "universal-chat"
        self.available_providers = []
        self.best_providers = {}
        self.fallback_mode = False

        # Initialize the AI detection and routing
        self._initialize_router()

    def _initialize_router(self):
        """Initialize LiteLLM Router with detected providers."""
        if not LITELLM_AVAILABLE:
            logger.warning("LiteLLM not available, using fallback mode")
            self.fallback_mode = True
            return

        try:
            # Detect available providers
            detector = AIDetector()
            import asyncio

            # Run detection in sync context
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If already in async context, create new loop for this
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, detector.detect_all_providers()
                    )
                    self.available_providers = future.result()
            else:
                self.available_providers = loop.run_until_complete(
                    detector.detect_all_providers()
                )

            self.best_providers = detector.get_best_providers()

            # Build model list for router
            model_list = self._build_model_list()

            if not model_list:
                logger.warning("No AI providers detected, using fallback mode")
                self.fallback_mode = True
                return

            # Initialize router with detected providers
            self.router = Router(
                model_list=model_list,
                fallbacks=self._build_fallbacks(),
                routing_strategy=os.getenv("ROUTING_STRATEGY", "latency-based-routing"),
                num_retries=int(os.getenv("MAX_RETRIES", "3")),
                timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
                cooldown_time=float(os.getenv("COOLDOWN_TIME", "60")),
            )

            # Set up caching
            self._setup_caching()

            logger.info(
                f"Initialized router with {len(model_list)} models from {len(self.available_providers)} providers"
            )

        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            self.fallback_mode = True

    def _build_model_list(self) -> List[Dict[str, Any]]:
        """Build model list for LiteLLM Router from detected providers."""
        model_list = []

        # Add embedding models
        embedding_providers = []
        if self.best_providers.get("local_embedding"):
            embedding_providers.append(self.best_providers["local_embedding"])
        if self.best_providers.get("cloud_embedding"):
            embedding_providers.append(self.best_providers["cloud_embedding"])

        for i, provider in enumerate(embedding_providers):
            if provider.embedding_models:
                # Use the first available embedding model
                model = provider.embedding_models[0]
                model_config = {
                    "model_name": f"{self.embedding_model_name}-{i}",
                    "litellm_params": self._get_litellm_params(provider, model),
                }
                model_list.append(model_config)

        # Add chat models
        chat_providers = []
        if self.best_providers.get("local_chat"):
            chat_providers.append(self.best_providers["local_chat"])
        if self.best_providers.get("cloud_chat"):
            chat_providers.append(self.best_providers["cloud_chat"])

        for i, provider in enumerate(chat_providers):
            if provider.chat_models:
                # Use the first available chat model
                model = provider.chat_models[0]
                model_config = {
                    "model_name": f"{self.chat_model_name}-{i}",
                    "litellm_params": self._get_litellm_params(provider, model),
                }
                model_list.append(model_config)

        return model_list

    def _get_litellm_params(self, provider: AIProvider, model: str) -> Dict[str, Any]:
        """Get LiteLLM parameters for a provider and model."""
        params = {}

        if provider.type == "local":
            # Local providers (LM Studio, Ollama, etc.)
            if provider.name == "ollama":
                params["model"] = f"ollama/{model}"
                params["api_base"] = provider.api_base
            else:
                # LM Studio and other OpenAI-compatible
                params["model"] = f"openai/{model}"
                params["api_base"] = f"{provider.api_base}/v1"
                params["api_key"] = "dummy"  # Local servers often don't need keys
        else:
            # Cloud providers
            if provider.name == "openai":
                params["model"] = f"openai/{model}"
                params["api_key"] = provider.api_key
            elif provider.name == "anthropic":
                params["model"] = f"anthropic/{model}"
                params["api_key"] = provider.api_key
            elif provider.name == "google":
                params["model"] = f"vertex_ai/{model}"
                params["api_key"] = provider.api_key
            elif provider.name == "cohere":
                params["model"] = f"cohere/{model}"
                params["api_key"] = provider.api_key
            elif provider.name == "groq":
                params["model"] = f"groq/{model}"
                params["api_key"] = provider.api_key
            else:
                # Default to OpenAI-compatible format
                params["model"] = f"openai/{model}"
                params["api_base"] = provider.api_base
                params["api_key"] = provider.api_key

        return params

    def _build_fallbacks(self) -> List[Dict[str, List[str]]]:
        """Build fallback chains for router."""
        fallbacks = []

        # Embedding fallbacks: local -> cloud -> offline
        embedding_fallback_chain = []
        if self.best_providers.get("local_embedding"):
            embedding_fallback_chain.append(f"{self.embedding_model_name}-0")
        if self.best_providers.get("cloud_embedding"):
            idx = 1 if self.best_providers.get("local_embedding") else 0
            embedding_fallback_chain.append(f"{self.embedding_model_name}-{idx}")

        if len(embedding_fallback_chain) > 1:
            fallbacks.append(
                {embedding_fallback_chain[0]: embedding_fallback_chain[1:]}
            )

        # Chat fallbacks: local -> cloud
        chat_fallback_chain = []
        if self.best_providers.get("local_chat"):
            chat_fallback_chain.append(f"{self.chat_model_name}-0")
        if self.best_providers.get("cloud_chat"):
            idx = 1 if self.best_providers.get("local_chat") else 0
            chat_fallback_chain.append(f"{self.chat_model_name}-{idx}")

        if len(chat_fallback_chain) > 1:
            fallbacks.append({chat_fallback_chain[0]: chat_fallback_chain[1:]})

        return fallbacks

    def _setup_caching(self):
        """Setup caching for LiteLLM - disk only, no Redis."""
        try:
            # Use disk-based cache only - no Redis dependency
            litellm.cache = litellm.Cache(
                type="disk", ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))
            )
            logger.debug("LiteLLM disk cache configured")

        except Exception as e:
            logger.debug(f"Failed to setup LiteLLM caching: {e}")
            # Disable caching if it fails
            litellm.cache = None

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using best available provider with fallbacks."""
        if self.fallback_mode or not self.router:
            return self._fallback_embedding(text)

        try:
            # Truncate text if too long
            text = text[:8000] if len(text) > 8000 else text

            # Use router for embedding
            response = await self.router.aembedding(
                model=self.embedding_model_name, input=text
            )

            return response.data[0]["embedding"]

        except Exception as e:
            logger.warning(f"Embedding generation failed, using fallback: {e}")
            return self._fallback_embedding(text)

    async def generate_tags(self, content: str) -> List[str]:
        """Generate tags using best available chat model with fallbacks."""
        if self.fallback_mode or not self.router:
            return self._fallback_tags(content)

        try:
            # Truncate content for tagging
            content_sample = content[:1000] if len(content) > 1000 else content

            prompt = f"""Analyze this content and generate 3-5 relevant tags that capture its main topics and themes.

Content: {content_sample}

Return only a JSON array of strings, nothing else. Example: ["python", "web development", "tutorial"]

Tags:"""

            response = await self.router.acompletion(
                model=self.chat_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )

            tags_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                tags = json.loads(tags_text)
                if isinstance(tags, list):
                    return [str(tag).lower().strip() for tag in tags[:5]]
            except json.JSONDecodeError:
                return self._extract_tags_from_text(tags_text)

        except Exception as e:
            logger.warning(f"Tag generation failed, using fallback: {e}")

        return self._fallback_tags(content)

    async def should_store(self, content: str) -> bool:
        """Determine if content is worth storing (for agent mode)."""
        # Get environment-based configuration
        if not os.getenv("AGENT_AUTO_CAPTURE", "true").lower() == "true":
            return True  # Store everything if auto-capture is disabled

        min_length = int(os.getenv("AGENT_MIN_CONTENT_LENGTH", "50"))
        max_length = int(os.getenv("AGENT_MAX_CONTENT_LENGTH", "10000"))

        # Length checks
        if len(content) < min_length or len(content) > max_length:
            return False

        content_lower = content.lower()

        # Skip uninteresting patterns
        skip_patterns = [
            "loading...",
            "please wait",
            "error 404",
            "page not found",
            "click here",
            "subscribe now",
            "cookie policy",
        ]

        if any(pattern in content_lower for pattern in skip_patterns):
            return False

        # Value indicators based on environment config
        valuable_patterns = []

        if os.getenv("AGENT_FILTER_CODE", "true").lower() == "true":
            valuable_patterns.extend(
                [
                    "def ",
                    "function ",
                    "class ",
                    "import ",
                    "```",
                    "git ",
                    "npm ",
                    "pip ",
                    "docker",
                    "kubectl",
                ]
            )

        if os.getenv("AGENT_FILTER_DOCS", "true").lower() == "true":
            valuable_patterns.extend(
                [
                    "how to",
                    "tutorial",
                    "guide",
                    "steps:",
                    "example:",
                    "solution",
                    "fix",
                    "problem",
                    "issue",
                    "documentation",
                ]
            )

        if os.getenv("AGENT_FILTER_CONFIGS", "true").lower() == "true":
            valuable_patterns.extend(
                [
                    "api",
                    "config",
                    "setup",
                    "installation",
                    "usage",
                    "environment",
                    "deployment",
                    "configuration",
                ]
            )

        valuable_score = sum(
            1 for pattern in valuable_patterns if pattern in content_lower
        )

        # Store if it has multiple valuable indicators
        return valuable_score >= 2

    def _fallback_embedding(self, text: str) -> List[float]:
        """Hash-based fallback embedding."""
        # Create deterministic embedding from text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Extend to 1536 dimensions (OpenAI compatibility)
        embedding = []
        for i in range(1536):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            embedding.append((byte_val - 128) / 128.0)

        return embedding

    def _fallback_tags(self, content: str) -> List[str]:
        """Rule-based tag generation fallback."""
        content_lower = content.lower()
        tags = []

        # Programming languages
        lang_patterns = {
            "python": ["python", "def ", "import ", ".py", "pip "],
            "javascript": ["javascript", "function ", "const ", "let ", ".js", "npm "],
            "typescript": ["typescript", "interface ", "type ", ".ts"],
            "java": ["java", "public class", "import java"],
            "c++": ["c++", "#include", "std::", "cout"],
            "rust": ["rust", "fn ", "let mut", "cargo"],
            "go": ["golang", "func ", "package ", 'import "'],
        }

        for lang, patterns in lang_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(lang)

        # Technologies
        tech_patterns = {
            "docker": ["docker", "dockerfile", "container"],
            "kubernetes": ["kubernetes", "kubectl", "k8s"],
            "git": ["git ", "github", "gitlab", "commit"],
            "database": ["sql", "mysql", "postgresql", "mongodb"],
            "web": ["http", "html", "css", "api", "rest"],
            "cloud": ["aws", "azure", "gcp", "cloud"],
        }

        for tech, patterns in tech_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(tech)

        # Content types
        if any(word in content_lower for word in ["tutorial", "guide", "how to"]):
            tags.append("tutorial")

        if any(word in content_lower for word in ["error", "bug", "fix", "issue"]):
            tags.append("troubleshooting")

        if any(word in content_lower for word in ["config", "setup", "install"]):
            tags.append("configuration")

        return list(set(tags))[:5]

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract tags from malformed LLM response."""
        tags = []

        # Try to find quoted strings
        quoted_pattern = r'"([^"]+)"'
        quotes = re.findall(quoted_pattern, text)
        if quotes:
            tags.extend([tag.strip().lower() for tag in quotes])

        # Try comma-separated values
        if not tags:
            clean_text = re.sub(r"[^\w\s,]", "", text)
            potential_tags = [word.strip().lower() for word in clean_text.split(",")]
            tags.extend([tag for tag in potential_tags if tag and 2 < len(tag) < 20])

        return tags[:5] if tags else ["untagged"]

    def get_provider_status(self) -> Dict[str, Any]:
        """Get current provider status and configuration."""
        if self.fallback_mode:
            return {
                "mode": "fallback",
                "providers": [],
                "message": "No AI providers available, using offline mode",
            }

        return {
            "mode": "router",
            "providers": [
                {
                    "name": p.name,
                    "type": p.type,
                    "available": p.available,
                    "latency_ms": p.latency_ms,
                    "models": len(p.models) if p.models else 0,
                }
                for p in self.available_providers
            ],
            "best_embedding": (
                self.best_providers.get("embedding", {}).name
                if self.best_providers.get("embedding")
                else None
            ),
            "best_chat": (
                self.best_providers.get("chat", {}).name
                if self.best_providers.get("chat")
                else None
            ),
            "router_configured": self.router is not None,
        }


# Maintain compatibility with existing code
ClipIntelligence = UniversalIntelligence
