"""
Qwen3-TTS VoiceDesign backend — synthesize with a voice created from a
natural-language description ("a gravelly noir detective, mid-50s, tired
warmth under the cynicism"), no reference audio required.

Consumes the ``designed`` profile type that the profile service already
serializes (``{"voice_type": "designed", "design_prompt": ...}``) — this
backend is the missing consumer of that stub.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import torch

from . import LANGUAGE_CODE_TO_NAME
from .base import (
    is_model_cached,
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# VoiceDesign ships as 1.7B only.
QWEN_VD_HF_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


class QwenVoiceDesignBackend:
    """Qwen3-TTS VoiceDesign — voice-from-description synthesis."""

    # Caller-overridable HF-generate sampling params, clamped to safe ranges.
    _PARAM_SPEC = {
        "temperature": (0.05, 2.0, float),
        "top_k": (1, 500, int),
        "top_p": (0.1, 1.0, float),
        "repetition_penalty": (1.0, 3.0, float),
        "max_new_tokens": (256, 8192, int),
    }

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = "1.7B"
        self.device = get_torch_device(allow_xpu=True, allow_directml=True)
        self._current_model_size: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: Optional[str] = None) -> str:
        return QWEN_VD_HF_REPO

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        return is_model_cached(self._get_model_path(model_size))

    async def load_model_async(self, model_size: Optional[str] = None) -> None:
        if self.model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    # Alias for compatibility with the TTSBackend protocol
    load_model = load_model_async

    def _load_model_sync(self) -> None:
        model_name = "qwen-voice-design-1.7B"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            from qwen_tts import Qwen3TTSModel

            model_path = self._get_model_path()
            logger.info("Loading Qwen VoiceDesign on %s...", self.device)

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )

        self._current_model_size = "1.7B"
        logger.info("Qwen VoiceDesign loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen VoiceDesign unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """VoiceDesign doesn't use reference audio. Designed profiles get
        their voice_prompt from the profile service; this fallback exists
        only for protocol compatibility."""
        return {
            "voice_type": "designed",
            "design_prompt": "A clear, natural narrator voice.",
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio with a designed voice.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with design_prompt (the voice description)
            language: Language code
            seed: Random seed for reproducibility
            instruct: Optional per-line delivery direction — appended to the
                voice description so one designed voice can still be directed
            params: Optional sampling overrides

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        from . import clamp_params

        await self.load_model_async(None)

        design_prompt = (voice_prompt.get("design_prompt") or "").strip()
        if not design_prompt:
            raise ValueError("Designed voice is missing its description (design_prompt)")

        # Per-line direction rides the same instruction channel as the design.
        full_instruct = design_prompt
        if instruct and instruct.strip():
            full_instruct = f"{design_prompt}. Delivery: {instruct.strip()}"

        gen_opts = clamp_params(params, self._PARAM_SPEC)

        def _generate_sync():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            logger.info("[VoiceDesign] Generating (%s)", language)

            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                instruct=full_instruct,
                language=lang_name.capitalize() if lang_name != "auto" else "Auto",
                **gen_opts,
            )
            return wavs[0], sample_rate

        audio, sample_rate = await asyncio.to_thread(_generate_sync)
        return audio, sample_rate
