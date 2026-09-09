"""Remote-Sensing Visual Question Answering (VQA) Specialist.

Implements BaseSpecialist for single-image VQA queries over Earth-observation imagery.
Isolates model-specific preprocessing, vision tokenization, generation, and VRAM cleanup.
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch

from backend.agent.schema import (
    EvidenceBundle,
    EvidenceImage,
    ModalityType,
    ModelCapability,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
)
from backend.models.base import BaseSpecialist
from backend.preprocessing.geotiff import GeoTIFFReader

logger = logging.getLogger(__name__)


class RemoteSensingVQASpecialist(BaseSpecialist):
    """Specialist vision-language model for remote-sensing question answering."""

    DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device=chosen_device)

        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.torch_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.model = None
        self.processor = None
        self._load_duration_ms = 0

    @property
    def capability(self) -> ModelCapability:
        """Declares the capabilities of this specialist."""
        return ModelCapability(
            identifier="RS_VQA",
            name=f"Remote-Sensing VQA ({self.model_id})",
            version="1.0.0",
            task=TaskType.VQA,
            supported_modalities=[ModalityType.OPTICAL, ModalityType.MULTISPECTRAL, ModalityType.SAR],
            supported_input_count=[1],
            requires_gpu=True,
            vram_budget_mb=4500,
            confidence_available=True,
            fallback_specialist_id=None,
        )

    def load(self) -> None:
        """Loads vision-language model and processor onto device."""
        if self._is_loaded:
            return

        import os

        # Check for shared VLM runtime toggle (Candidate 1: enabled by default, 100% reversible)
        use_shared = os.environ.get("SATQUERY_SHARED_VLM_RUNTIME", "1") != "0"
        if use_shared:
            from backend.models.shared_vlm import get_shared_vlm
            handle = get_shared_vlm(
                model_id=self.model_id,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            self.processor = handle.processor
            self.model = handle.model
            self._is_adapted = handle.is_adapted
            self._using_shared_runtime = True
            self._load_duration_ms = handle.load_duration_ms
            self._is_loaded = True
            logger.info("VQA specialist attached to Shared VLM Runtime (adapted=%s) in %d ms", self._is_adapted, self._load_duration_ms)
            return

        start_time = time.perf_counter()
        logger.info("Loading dedicated VQA model [%s] onto device [%s]...", self.model_id, self.device)

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model with flash attention / sdpa
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if self.device != "cuda" and hasattr(self.model, "to"):
            self.model.to(self.device)

        # Check for adapted model toggle (SATQUERY_USE_ADAPTED_VLM=1)
        import os
        use_adapted = os.environ.get("SATQUERY_USE_ADAPTED_VLM", "0") == "1"
        adapter_path = Path("models/adapters/qwen2_vl_rs_lora")
        if use_adapted and adapter_path.exists():
            from peft import PeftModel
            logger.info("Loading RS-adapted LoRA adapter from %s...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self._is_adapted = True
        else:
            self._is_adapted = False

        self.model.eval()
        self._is_loaded = True
        self._load_duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("VQA model loaded successfully (adapted=%s) in %d ms", self._is_adapted, self._load_duration_ms)

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        """Executes VQA inference on input raster image and query."""
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        query = inputs.query or "Describe the predominant features and land use in this satellite image."
        image_path = Path(inputs.primary_image_path)

        # 1. Preprocessing: Safely read raster into normalized RGB [H, W, 3]
        rgb_array, raw_meta = GeoTIFFReader.read_normalized_rgb(image_path)
        pil_image = Image.fromarray(rgb_array)

        # 2. Build remote-sensing analytical prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert remote-sensing intelligence analyst. You analyze overhead Earth observation, "
                    "aerial, and satellite imagery. Provide precise, factual answers grounded in the visual evidence "
                    "such as land cover, infrastructure, spectral characteristics, and terrain."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query},
                ],
            },
        ]

        # 3. Format prompt with chat template
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [pil_image]

        inputs_tensor = self.processor(
            text=[prompt_text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs_tensor = inputs_tensor.to("cuda")

        max_new_tokens = int(inputs.parameters.get("max_new_tokens", 128))
        temperature = float(inputs.parameters.get("temperature", 0.2))

        # 4. Generate prediction (Candidate 2: inference_mode)
        use_inference_mode = os.environ.get("SATQUERY_USE_INFERENCE_MODE", "1") == "1"
        context_mgr = torch.inference_mode() if use_inference_mode else torch.no_grad()
        with context_mgr:
            generated_ids = self.model.generate(
                **inputs_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Strip input prompt tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs_tensor.input_ids, generated_ids.sequences)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # 5. Compute generation confidence heuristic from output token log-probabilities
        confidence_score = 0.85  # baseline
        if generated_ids.scores:
            try:
                # Softmax top token likelihoods
                token_probs = []
                for score in generated_ids.scores:
                    probs = torch.softmax(score, dim=-1)
                    max_prob = torch.max(probs).item()
                    token_probs.append(max_prob)
                if token_probs:
                    confidence_score = float(np.mean(token_probs))
            except Exception:
                confidence_score = 0.80

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # 6. Save Preview Artifact (PNG preview for web display)
        static_dir = Path("backend/static/previews")
        static_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"vqa_{int(time.time() * 1000) % 100000}"
        preview_filename = f"primary_preview_{run_id}.png"
        pil_image.save(static_dir / preview_filename, format="PNG")

        # Build evidence bundle
        evidence = EvidenceBundle(
            images=[
                EvidenceImage(
                    role="primary",
                    url=f"/api/v1/static/previews/{preview_filename}",
                    width=raw_meta["width"],
                    height=raw_meta["height"],
                    crs=raw_meta.get("crs"),
                    bounds=raw_meta.get("bounds"),
                )
            ],
            masks=[],  # VQA does not produce spatial masks; explicit empty list
            boxes=[],  # VQA does not produce bounding boxes; explicit empty list
            statistics=[],
            regions=[],
        )

        return SpecialistOutput(
            specialist_id=self.capability.identifier,
            success=True,
            answer_text=output_text,
            confidence=round(confidence_score, 4),
            evidence=evidence,
            parameters_used={
                "model_id": self.model_id,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "is_adapted": getattr(self, "_is_adapted", False),
                "adapter_path": "models/adapters/qwen2_vl_rs_lora" if getattr(self, "_is_adapted", False) else None,
            },
            warnings=[],
            execution_time_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        """Verifies operational readiness."""
        if self._is_loaded:
            return self.model is not None and self.processor is not None
        return True  # Ready to be loaded

    def cleanup(self) -> None:
        """Releases CUDA memory buffers."""
        if getattr(self, "_using_shared_runtime", False):
            from backend.models.shared_vlm import release_shared_vlm
            release_shared_vlm()
            self.model = None
            self.processor = None
            self._is_loaded = False
            self._using_shared_runtime = False
            logger.info("VQA detached from Shared VLM Runtime.")
            return

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VQA specialist memory released.")
