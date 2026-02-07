"""Visual Language Model interface for screenshot analysis."""

import logging
from io import BytesIO
from typing import Optional

from PIL import Image, ImageChops

from major_tom.config import Config
from major_tom.constants import DIFF_RESIZE_SIZE, VLM_MAX_DIMENSION
from major_tom.llm.base import LLMBackend
from major_tom.memory.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class VisualHarvester:
    """Analyzes screenshots using a Vision Language Model."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self._llm = llm_backend
        self.last_thumb: Optional[Image.Image] = None
        self.diff_threshold = Config.VISUAL_DIFF_THRESHOLD
        self.audit = audit_logger

    def harvest(self, task_prompt: str, screenshot_image: Image.Image) -> str:
        """Analyze screenshot using VLM."""
        if screenshot_image is None:
            return "Error: No image."

        try:
            current_thumb = screenshot_image.resize(
                (DIFF_RESIZE_SIZE, DIFF_RESIZE_SIZE)
            ).convert("L")

            if self.last_thumb:
                diff = ImageChops.difference(current_thumb, self.last_thumb)
                diff_hist = diff.histogram()
                total_pixels = DIFF_RESIZE_SIZE * DIFF_RESIZE_SIZE
                unchanged_pixels = diff_hist[0]
                similarity_ratio = unchanged_pixels / total_pixels

                if similarity_ratio > self.diff_threshold:
                    if self.audit:
                        self.audit.log(
                            component="Eye",
                            event_type="STATIC_SKIP",
                            data={
                                "task_prompt": task_prompt,
                                "similarity_ratio": f"{similarity_ratio:.4f}",
                                "threshold": f"{self.diff_threshold:.4f}",
                                "reason": "Screen unchanged - skipping VLM call",
                                "image_size": f"{screenshot_image.size}",
                            },
                        )
                    return "[STATIC] Screen unchanged."

            self.last_thumb = current_thumb

            img_to_send = screenshot_image.copy()
            if max(img_to_send.size) > VLM_MAX_DIMENSION:
                img_to_send.thumbnail((VLM_MAX_DIMENSION, VLM_MAX_DIMENSION))

            img_byte_arr = BytesIO()
            img_to_send.convert("RGB").save(img_byte_arr, format="JPEG", quality=85)

            full_prompt = f"{Config.VLM_SYSTEM_PROMPT}\n[Focus]: {task_prompt}"

            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_CALL_START",
                    data={
                        "model": Config.EYE_MODEL,
                        "task_prompt": task_prompt,
                        "full_prompt": full_prompt,
                        "image_size": f"{img_to_send.size}",
                        "image_bytes": len(img_byte_arr.getvalue()),
                    },
                )

            res = self._llm.generate(
                model=Config.EYE_MODEL,
                prompt=full_prompt,
                images=[img_byte_arr.getvalue()],
                stream=False,
                keep_alive="5m",
            )

            raw_response = res.text.strip().replace("\n", " ")

            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_CALL_COMPLETE",
                    data={
                        "model": Config.EYE_MODEL,
                        "task_prompt": task_prompt,
                        "raw_response": raw_response,
                        "input_tokens": res.prompt_tokens,
                        "output_tokens": res.completion_tokens,
                        "total_tokens": res.total_tokens,
                    },
                )

            return raw_response

        except Exception as e:
            error_msg = f"Visual Error: {e}"
            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_ERROR",
                    data={
                        "task_prompt": task_prompt,
                        "error": str(e),
                        "model": Config.EYE_MODEL,
                    },
                )
            return error_msg
