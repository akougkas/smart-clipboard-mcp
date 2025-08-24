"""Data models for Smart Clipboard MCP Server."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class Clip(BaseModel):
    """Simple and efficient clip model."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result with similarity score."""

    id: str
    content: str
    tags: List[str]
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClipResponse(BaseModel):
    """Response for clip operations."""

    id: str
    status: str
    message: Optional[str] = None
