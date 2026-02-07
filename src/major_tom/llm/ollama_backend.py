"""Ollama LLM backend implementation."""

import logging
import time
from typing import Any, Dict, List, Optional

import ollama

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Wraps ollama.generate() and ollama.embeddings() with standardized responses."""

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
        kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if images is not None:
            kwargs["images"] = images
        if format is not None:
            kwargs["format"] = format
        if options is not None:
            kwargs["options"] = options
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive

        start = time.monotonic()
        res = ollama.generate(**kwargs)
        latency = (time.monotonic() - start) * 1000

        prompt_tokens = res.get("prompt_eval_count", 0)
        completion_tokens = res.get("eval_count", 0)

        return LLMResponse(
            text=res["response"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            latency_ms=latency,
            raw=res,
        )

    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        start = time.monotonic()
        res = ollama.embeddings(model=model, prompt=prompt)
        latency = (time.monotonic() - start) * 1000

        vector = res.get("embedding", [])
        # Estimate tokens: ~4 chars per token for English
        estimated_tokens = max(1, len(prompt) // 4)

        return EmbeddingResponse(
            vector=vector,
            prompt_tokens=estimated_tokens,
            model=model,
            latency_ms=latency,
        )
