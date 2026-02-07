"""Mock LLM backend for testing."""

import threading
from typing import Any, Dict, List, Optional

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    """Configurable mock backend for deterministic testing."""

    def __init__(self):
        self._generate_response = LLMResponse(
            text='{"action": "SKIP", "reason": "mock", "next_check_delay": 5}',
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="mock",
            latency_ms=1.0,
        )
        self._embed_response = EmbeddingResponse(
            vector=[0.0] * 768,
            prompt_tokens=5,
            model="mock",
            latency_ms=0.5,
        )
        self._generate_count = 0
        self._embed_count = 0
        self._lock = threading.Lock()
        self._generate_calls: List[Dict[str, Any]] = []
        self._embed_calls: List[Dict[str, Any]] = []

    def set_generate_response(self, response: LLMResponse) -> None:
        """Configure the canned generate response."""
        self._generate_response = response

    def set_embedding_response(self, response: EmbeddingResponse) -> None:
        """Configure the canned embedding response."""
        self._embed_response = response

    @property
    def generate_count(self) -> int:
        return self._generate_count

    @property
    def embed_count(self) -> int:
        return self._embed_count

    @property
    def generate_calls(self) -> List[Dict[str, Any]]:
        return list(self._generate_calls)

    @property
    def embed_calls(self) -> List[Dict[str, Any]]:
        return list(self._embed_calls)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[bytes]] = None,
        format: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> LLMResponse:
        with self._lock:
            self._generate_count += 1
            self._generate_calls.append({
                "model": model,
                "prompt": prompt,
                "images": images,
                "format": format,
                "options": options,
            })
        resp = LLMResponse(
            text=self._generate_response.text,
            prompt_tokens=self._generate_response.prompt_tokens,
            completion_tokens=self._generate_response.completion_tokens,
            total_tokens=self._generate_response.total_tokens,
            model=model,
            latency_ms=self._generate_response.latency_ms,
            raw=self._generate_response.raw,
        )
        return resp

    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        with self._lock:
            self._embed_count += 1
            self._embed_calls.append({"model": model, "prompt": prompt})
        return EmbeddingResponse(
            vector=list(self._embed_response.vector),
            prompt_tokens=self._embed_response.prompt_tokens,
            model=model,
            latency_ms=self._embed_response.latency_ms,
        )

    def reset(self) -> None:
        """Reset all counters and call history."""
        with self._lock:
            self._generate_count = 0
            self._embed_count = 0
            self._generate_calls.clear()
            self._embed_calls.clear()
