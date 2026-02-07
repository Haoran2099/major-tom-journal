"""LLM abstraction layer."""

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse

__all__ = ["LLMBackend", "LLMResponse", "EmbeddingResponse"]
