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

    def _compose_prompt(self, task_prompt: str) -> str:
        """Build final VLM prompt while preserving structured focus capsules."""
        cleaned = (task_prompt or "Analyze current visual activity").strip()
        if "[Focus]" in cleaned:
            return f"{Config.VLM_SYSTEM_PROMPT}\n{cleaned}"
        return f"{Config.VLM_SYSTEM_PROMPT}\n[Focus]: {cleaned}"

    def harvest(self, task_prompt: str, screenshot_image: Image.Image) -> str:
        """Analyze screenshot using VLM."""
        if screenshot_image is None:
            return "Error: No image."

        try:
            # Use high-quality downsampling to preserve UI details (text/icons)
            resample_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            current_thumb = screenshot_image.resize(
                (DIFF_RESIZE_SIZE, DIFF_RESIZE_SIZE), resample=resample_method
            ).convert("L")

            if self.last_thumb:
                # 1. Structural difference using pixel diff on larger thumbnail
                diff = ImageChops.difference(current_thumb, self.last_thumb)
                diff_hist = diff.histogram()
                total_pixels = DIFF_RESIZE_SIZE * DIFF_RESIZE_SIZE
                
                # Count pixels that changed significantly (>10 intensity difference)
                # This ignores minor compression artifacts
                significant_changes = sum(diff_hist[10:]) 
                change_ratio = significant_changes / total_pixels
                
                # Inverse logic: similarity = 1 - change_ratio
                similarity_ratio = 1.0 - change_ratio

                if similarity_ratio > self.diff_threshold:
                    if self.audit:
                        self.audit.log(
                            component="Eye",
                            event_type="STATIC_SKIP",
                            data={
                                "task_prompt": task_prompt,
                                "similarity_ratio": f"{similarity_ratio:.4f}",
                                "threshold": f"{self.diff_threshold:.4f}",
                                "reason": "Screen unchanged (structural check) - skipping VLM call",
                                "image_size": f"{screenshot_image.size}",
                            },
                        )
                    return "[STATIC] Screen likely unchanged."

            self.last_thumb = current_thumb

            img_to_send = screenshot_image.copy()
            if max(img_to_send.size) > VLM_MAX_DIMENSION:
                img_to_send.thumbnail((VLM_MAX_DIMENSION, VLM_MAX_DIMENSION))

            img_byte_arr = BytesIO()
            img_to_send.convert("RGB").save(img_byte_arr, format="JPEG", quality=85)

            full_prompt = self._compose_prompt(task_prompt)

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
