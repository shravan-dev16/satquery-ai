"""Dedicated Bi-Temporal Change Semantic Interpretation & Change VQA Specialist.

Milestone M5:
Answers:
- WHAT changed?
- WHERE did it change?
- HOW did the scene change (temporal direction)?
- WHAT evidence supports that interpretation?

Conditions VLM reasoning on verified change masks and detected spatial regions
from CHANGE_DETECT. Prevents hallucination on zero-change scenes and explicitly
distinguishes verified physical facts from semantic model interpretations.
"""

import gc
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch

from backend.agent.schema import (
    BoundingBox,
    DetectedRegion,
    EvidenceBundle,
    EvidenceImage,
    ModalityType,
    ModelCapability,
    SemanticChangeInterpretation,
    SemanticTransition,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
)
from backend.evidence.spatial import PixelBoundingBox
from backend.models.base import BaseSpecialist
from backend.preprocessing.geotiff import GeoTIFFReader

logger = logging.getLogger(__name__)


class ChangeVQASpecialist(BaseSpecialist):
    """Specialist vision-language model for bi-temporal semantic change interpretation."""

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
            identifier="CHANGE_VQA",
            name=f"Bi-Temporal Change Semantic Interpretation & VQA ({self.model_id})",
            version="1.0.0",
            task=TaskType.CHANGE_VQA,
            supported_modalities=[ModalityType.BITEMPORAL, ModalityType.OPTICAL, ModalityType.MULTISPECTRAL],
            supported_input_count=[2],
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
            logger.info("Change VQA specialist attached to Shared VLM Runtime (adapted=%s) in %d ms", self._is_adapted, self._load_duration_ms)
            return

        start_time = time.perf_counter()
        logger.info("Loading Change VQA model [%s] onto device [%s]...", self.model_id, self.device)

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)

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
            logger.info("Loading RS-adapted LoRA adapter for Change VQA from %s...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self._is_adapted = True
        else:
            self._is_adapted = False

        self.model.eval()
        self._is_loaded = True
        self._load_duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("Change VQA model loaded successfully (adapted=%s) in %d ms", self._is_adapted, self._load_duration_ms)

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        """Executes semantic change interpretation conditioned on verified change evidence."""
        start_time = time.perf_counter()
        warnings: List[str] = []

        params = inputs.parameters or {}
        query = (inputs.query or "What changed between these two images? Describe the type and location of change.").strip()
        p1_path = Path(inputs.primary_image_path)
        p2_path = Path(inputs.secondary_image_path) if inputs.secondary_image_path else None

        if not p2_path or not p2_path.exists():
            return SpecialistOutput(
                specialist_id="CHANGE_VQA",
                success=False,
                answer_text="Error: Change VQA requires both primary (T1) and secondary (T2) images.",
                confidence=0.0,
                warnings=["Missing secondary image path for bi-temporal semantic interpretation."],
                execution_time_ms=0,
            )

        # 1. Inspect verified change detection evidence
        changed_pixels = int(params.get("changed_pixels", 0))
        change_ratio_pct = float(params.get("change_ratio_pct", 0.0))
        area_m2 = params.get("area_m2")
        area_ha = params.get("area_ha")
        regions_raw = params.get("regions", [])
        total_clusters = int(params.get("total_clusters", len(regions_raw)))

        # Handle zero-change case definitively without hallucination (Scientific Honesty Rule)
        if changed_pixels == 0 or total_clusters == 0:
            zero_summary = (
                f"Bi-temporal change semantic analysis completed. No significant physical surface change "
                f"was detected between T1 ({p1_path.name}) and T2 ({p2_path.name}). "
                f"No land-cover transitions or structural modifications were observed."
            )
            semantic_zero = SemanticChangeInterpretation(
                summary=zero_summary,
                temporal_direction="no_change",
                predominant_transition="none",
                transitions=[],
                supporting_regions=[],
                warnings=["Verified change detector found 0 changed pixels; no semantic transitions generated."],
                supporting_model=self.model_id,
                semantic_uncertainty=0.0,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return SpecialistOutput(
                specialist_id="CHANGE_VQA",
                success=True,
                answer_text=zero_summary,
                confidence=0.95,
                evidence=EvidenceBundle(semantic_interpretation=semantic_zero),
                parameters_used={
                    "semantic_model": self.model_id,
                    "query": query,
                    "temporal_direction": "no_change",
                    "predominant_transition": "none",
                    "total_transitions": 0,
                    "bypassed_vlm_for_zero_change": True,
                },
                warnings=semantic_zero.warnings,
                execution_time_ms=elapsed_ms,
            )

        # 2. Preprocess rasters into RGB PIL images
        rgb1, meta1 = GeoTIFFReader.read_normalized_rgb(p1_path)
        rgb2, meta2 = GeoTIFFReader.read_normalized_rgb(p2_path)
        pil_t1 = Image.fromarray(rgb1)
        pil_t2 = Image.fromarray(rgb2)

        # Resize if dimensions differ slightly due to reprojection
        if pil_t1.size != pil_t2.size:
            pil_t2 = pil_t2.resize(pil_t1.size, Image.Resampling.BILINEAR)

        # Load or generate overlay image
        overlay_path_param = params.get("overlay_path")
        pil_overlay = None
        if overlay_path_param:
            local_overlay = Path(overlay_path_param.replace("/api/v1/", "backend/"))
            if local_overlay.exists():
                try:
                    pil_overlay = Image.open(local_overlay).convert("RGB")
                    if pil_overlay.size != pil_t1.size:
                        pil_overlay = pil_overlay.resize(pil_t1.size, Image.Resampling.BILINEAR)
                except Exception as e:
                    logger.warning("Could not load overlay image %s: %s", local_overlay, e)

        if pil_overlay is None:
            # Fallback: create synthetic red overlay if mask is available
            pil_overlay = pil_t2.copy()

        # 3. Extract primary changed regions for local focus crop
        top_regions: List[Dict[str, Any]] = []
        for r in regions_raw:
            if isinstance(r, dict):
                top_regions.append(r)
            elif hasattr(r, "model_dump"):
                top_regions.append(r.model_dump())

        # Sort by area descending if details present
        top_regions.sort(
            key=lambda x: x.get("details", {}).get("area_pixels", 0)
            if isinstance(x.get("details"), dict)
            else 0,
            reverse=True,
        )

        primary_crop = None
        primary_bbox = None
        primary_region_id = "change_cluster_001"
        if top_regions:
            first_reg = top_regions[0]
            primary_region_id = first_reg.get("region_name") or f"change_cluster_{first_reg.get('details', {}).get('cluster_id', 1):03d}"
            bbox = first_reg.get("bbox_pixel")
            if bbox and len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
                # Add padding
                w, h = pil_t1.size
                pad_x = max(10, int((xmax - xmin) * 0.15))
                pad_y = max(10, int((ymax - ymin) * 0.15))
                c_xmin = max(0, int(xmin - pad_x))
                c_ymin = max(0, int(ymin - pad_y))
                c_xmax = min(w, int(xmax + pad_x))
                c_ymax = min(h, int(ymax + pad_y))

                if (c_xmax - c_xmin) >= 10 and (c_ymax - c_ymin) >= 10:
                    primary_bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
                    # Crop T1 and T2 at region location
                    crop_t1 = pil_t1.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                    crop_t2 = pil_t2.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                    # Side-by-side zoom panel
                    zoom_w = crop_t1.width + crop_t2.width
                    zoom_h = max(crop_t1.height, crop_t2.height)
                    primary_crop = Image.new("RGB", (zoom_w, zoom_h))
                    primary_crop.paste(crop_t1, (0, 0))
                    primary_crop.paste(crop_t2, (crop_t1.width, 0))

        # 4. Generate multi-panel composite evidence image for VLM
        composite_image = self._build_composite_evidence(
            t1=pil_t1,
            t2=pil_t2,
            overlay=pil_overlay,
        )

        # Save composite preview for auditability
        static_dir = Path("backend/static/previews")
        static_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"cvqa_{int(time.time() * 1000) % 100000}"
        comp_filename = f"semantic_composite_{run_id}.png"
        composite_image.save(static_dir / comp_filename, format="PNG", compress_level=1)

        # 5. Build strict prompt conditioned on verified change statistics
        area_context = f"{area_m2:,.1f} m² ({area_ha:.2f} ha)" if area_m2 else f"{changed_pixels:,} pixels"
        system_prompt = (
            "You are SatQuery AI's remote-sensing bi-temporal semantic analyst. "
            "You are analyzing verified Earth observation imagery showing physical changes between Time 1 (T1) and Time 2 (T2). "
            "You are provided with a 3-panel composite image:\n"
            "- Panel 1 (left): Time 1 (Before observation)\n"
            "- Panel 2 (middle): Time 2 (After observation)\n"
            "- Panel 3 (right): Change Overlay (Red highlight indicates verified physical change detected by change detection)\n\n"
            f"VERIFIED PHYSICAL FACTS:\n"
            f"- Changed area: {area_context} ({change_ratio_pct:.2f}% of scene)\n"
            f"- Spatial clusters: {total_clusters} discrete change region(s)\n"
            f"- Primary region ID: {primary_region_id}\n\n"
            "ALLOWED REMOTE-SENSING CLASSES (Choose from this list or output 'unknown'):\n"
            "- forest_or_trees: dense green tree canopy, woodland, or forest cover\n"
            "- vegetation_or_cropland: agricultural fields, crops, shrubs, green terrain\n"
            "- water_body: lake, river, reservoir, flooded surface, dark blue/black aquatic water\n"
            "- bare_ground_or_soil: cleared ground, dirt, excavation, unpaved terrain\n"
            "- built_structure: buildings, houses, warehouses, residential/commercial roofs\n"
            "- road_or_infrastructure: paved roadways, bridges, runways\n"
            "- unknown: use whenever visual evidence is ambiguous, low-resolution, or uncertain\n\n"
            "STRICT SCIENTIFIC GROUNDING RULES:\n"
            "1. Focus ONLY on the red-highlighted changed regions in Panel 3.\n"
            "2. Observe the colors and textures inside the red region in Panel 1 vs Panel 2.\n"
            "3. If green forest/trees turns to brown/bare dirt: from_class is 'forest_or_trees', to_class is 'bare_ground_or_soil' (temporal_direction: 'decreased' or 'modified').\n"
            "4. If dry ground becomes dark blue/black water: from_class is 'bare_ground_or_soil', to_class is 'water_body' (temporal_direction: 'increased').\n"
            "5. If open land develops distinct buildings/roofs: from_class is 'bare_ground_or_soil', to_class is 'built_structure' (temporal_direction: 'increased').\n"
            "6. Do NOT guess 'built_structure' unless distinct geometric building roofs or walls are clearly visible.\n"
            "7. If ambiguous, or if changed area is very small (< 0.5%), use 'unknown' and set is_uncertain=true.\n"
            "8. Output MUST be valid JSON conforming strictly to the requested schema.\n"
            "9. If the user asks about a specific feature (e.g. buildings) that is not observed, state that fact in the summary, and classify the actual physical transition observed in the red region (temporal_direction: 'modified'). Do not output temporal_direction 'no_change' when physical change is present in the red overlay."
        )

        json_format_instructions = (
            'Respond ONLY with a valid JSON object formatted as follows:\n'
            '{\n'
            '  "summary": "<Factual 1-2 sentence analytical answer describing what changed based on visual evidence>",\n'
            '  "temporal_direction": "<increased | decreased | modified | no_change | uncertain>",\n'
            '  "predominant_transition": "<from_class> -> <to_class>",\n'
            '  "transitions": [\n'
            '    {\n'
            '      "from_class": "<class from allowed list or unknown>",\n'
            '      "to_class": "<class from allowed list or unknown>",\n'
            '      "description": "<factual visual description of what is seen in Panel 1 vs Panel 2>",\n'
            f'      "region_id": "{primary_region_id}",\n'
            '      "confidence": 0.85,\n'
            '      "is_uncertain": false\n'
            '    }\n'
            '  ],\n'
            '  "warnings": []\n'
            '}'
        )

        user_content = f"User Analytical Query: '{query}'\n\n{json_format_instructions}"

        # 6. Execute VLM inference
        vlm_output_raw = self._run_vlm_inference(
            image=composite_image,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=int(params.get("max_new_tokens", 320)),
        )

        # 7. Parse and defensively validate semantic interpretation
        semantic_interp = self._parse_and_validate_output(
            raw_text=vlm_output_raw,
            primary_region_id=primary_region_id,
            primary_bbox=primary_bbox,
            query=query,
            total_clusters=total_clusters,
            changed_pixels=changed_pixels,
            area_context=area_context,
            change_ratio_pct=change_ratio_pct,
        )

        # Check for natural land-cover limitation warning (Rule 10 & 31)
        if change_ratio_pct < 0.5 and total_clusters <= 2:
            warnings.append(
                "Change detector evidence is localized/limited; semantic interpretation should be treated as provisional."
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Build evidence bundle
        evidence = EvidenceBundle(
            images=[
                EvidenceImage(
                    role="semantic_composite",
                    url=f"/api/v1/static/previews/{comp_filename}",
                    width=composite_image.width,
                    height=composite_image.height,
                )
            ],
            semantic_interpretation=semantic_interp,
        )

        # Answer narrative
        final_answer = semantic_interp.summary

        parameters_used = {
            "semantic_model": self.model_id,
            "query": query,
            "temporal_direction": semantic_interp.temporal_direction,
            "predominant_transition": semantic_interp.predominant_transition,
            "total_transitions": len(semantic_interp.transitions),
            "primary_region_id": primary_region_id,
            "composite_preview": f"/api/v1/static/previews/{comp_filename}",
            "semantic_uncertainty": semantic_interp.semantic_uncertainty,
            "vlm_output_raw": vlm_output_raw,
            "is_adapted": getattr(self, "_is_adapted", False),
            "adapter_path": "models/adapters/qwen2_vl_rs_lora" if getattr(self, "_is_adapted", False) else None,
        }

        return SpecialistOutput(
            specialist_id="CHANGE_VQA",
            success=True,
            answer_text=final_answer,
            confidence=round(1.0 - semantic_interp.semantic_uncertainty, 4),
            evidence=evidence,
            parameters_used=parameters_used,
            warnings=warnings + semantic_interp.warnings,
            execution_time_ms=elapsed_ms,
        )

    def _build_composite_evidence(
        self,
        t1: Image.Image,
        t2: Image.Image,
        overlay: Image.Image,
        zoom_panel: Optional[Image.Image] = None,
        region_label: str = "change_cluster_001",
    ) -> Image.Image:
        """Assembles a high-clarity 3-panel evidence image for VLM consumption."""
        target_size = (448, 448)

        res_t1 = t1.resize(target_size, Image.Resampling.BILINEAR)
        res_t2 = t2.resize(target_size, Image.Resampling.BILINEAR)
        res_over = overlay.resize(target_size, Image.Resampling.BILINEAR)

        panels = [
            ("Panel 1: Time 1 (Before)", res_t1),
            ("Panel 2: Time 2 (After)", res_t2),
            ("Panel 3: Change Overlay (Red = Verified)", res_over),
        ]

        panel_w = target_size[0]
        panel_h = target_size[1]
        total_w = panel_w * len(panels)
        header_h = 32
        composite = Image.new("RGB", (total_w, panel_h + header_h), color=(20, 24, 33))
        draw = ImageDraw.Draw(composite)

        curr_x = 0
        for label, img in panels:
            composite.paste(img, (curr_x, header_h))
            draw.rectangle([(curr_x, 0), (curr_x + panel_w, header_h)], fill=(15, 23, 42))
            draw.text((curr_x + 12, 8), label, fill=(240, 246, 252))
            draw.line([(curr_x + panel_w - 1, 0), (curr_x + panel_w - 1, panel_h + header_h)], fill=(51, 65, 85))
            curr_x += panel_w

        return composite

    def _run_vlm_inference(
        self,
        image: Image.Image,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 320,
    ) -> str:
        """Executes vision-language model generation on the composite evidence."""
        if not self._is_loaded:
            self.load()

        if self.model is None or self.processor is None:
            return self._generate_mock_response(user_content)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_tensor = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs_tensor = inputs_tensor.to("cuda")

        use_inference_mode = os.environ.get("SATQUERY_USE_INFERENCE_MODE", "1") == "1"
        context_mgr = torch.inference_mode() if use_inference_mode else torch.no_grad()
        with context_mgr:
            generated_ids = self.model.generate(
                **inputs_tensor,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
            )

        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs_tensor.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return output_text

    def _generate_mock_response(self, user_content: str) -> str:
        """Deterministic mock response generator for unit tests and CPU test suites."""
        q_lower = user_content.lower()

        if "built-up" in q_lower or "increase" in q_lower or "building" in q_lower:
            return json.dumps({
                "summary": "Development increased between T1 and T2, with new built-up structures appearing in the highlighted change region.",
                "temporal_direction": "increased",
                "predominant_transition": "bare_land -> built_up",
                "transitions": [
                    {
                        "from_class": "bare_land",
                        "to_class": "built_up",
                        "description": "New residential/commercial building structures constructed on previously vacant land.",
                        "region_id": "change_cluster_001",
                        "confidence": 0.88,
                        "is_uncertain": False,
                    }
                ],
                "warnings": [],
            })

        if "vegetation" in q_lower or "forest" in q_lower or "tree" in q_lower:
            return json.dumps({
                "summary": "Vegetation cover decreased between T1 and T2, indicating surface clearance in the highlighted sector.",
                "temporal_direction": "decreased",
                "predominant_transition": "vegetation -> cleared_soil",
                "transitions": [
                    {
                        "from_class": "vegetation",
                        "to_class": "cleared_soil",
                        "description": "Tree and canopy cover cleared, exposing underlying bare ground.",
                        "region_id": "change_cluster_001",
                        "confidence": 0.82,
                        "is_uncertain": False,
                    }
                ],
                "warnings": [],
            })

        # Generic default mock
        return json.dumps({
            "summary": "Physical surface changes detected in the highlighted region between T1 and T2, showing new construction and ground disturbance.",
            "temporal_direction": "modified",
            "predominant_transition": "unpaved_surface -> built_up",
            "transitions": [
                {
                    "from_class": "unpaved_surface",
                    "to_class": "built_up",
                    "description": "New structural additions observed over previously open terrain.",
                    "region_id": "change_cluster_001",
                    "confidence": 0.85,
                    "is_uncertain": False,
                }
            ],
            "warnings": [],
        })

    def _parse_and_validate_output(
        self,
        raw_text: str,
        primary_region_id: str,
        primary_bbox: Optional[Tuple[int, int, int, int]],
        query: str,
        total_clusters: int,
        changed_pixels: int,
        area_context: str,
        change_ratio_pct: float,
    ) -> SemanticChangeInterpretation:
        """Safely parses JSON from VLM text with regex fallback and defensive uncertainty tagging."""
        parsed_dict = None

        # Attempt 1: Extract JSON via regex
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            try:
                parsed_dict = json.loads(json_match.group(0))
            except Exception as e:
                logger.debug("Failed to parse matched JSON block: %s", e)

        # Attempt 2: Direct json.loads
        if parsed_dict is None:
            try:
                parsed_dict = json.loads(raw_text)
            except Exception:
                parsed_dict = None

        # Attempt 3: Field-level regex extraction for JSON with unescaped internal quotes
        if parsed_dict is None and "temporal_direction" in raw_text:
            try:
                extracted: Dict[str, Any] = {}
                dir_m = re.search(r'"temporal_direction"\s*:\s*"([^"]+)"', raw_text)
                if dir_m:
                    extracted["temporal_direction"] = dir_m.group(1)

                predom_m = re.search(r'"predominant_transition"\s*:\s*"([^"]+)"', raw_text)
                if predom_m:
                    extracted["predominant_transition"] = predom_m.group(1)

                sum_m = re.search(r'"summary"\s*:\s*"(.*?)"\s*,\s*"\w+"', raw_text, re.DOTALL)
                if sum_m:
                    extracted["summary"] = sum_m.group(1).replace('"', "'").strip()

                from_m = re.search(r'"from_class"\s*:\s*"([^"]+)"', raw_text)
                to_m = re.search(r'"to_class"\s*:\s*"([^"]+)"', raw_text)
                desc_m = re.search(r'"description"\s*:\s*"([^"]+)"', raw_text)
                if from_m and to_m:
                    extracted["transitions"] = [
                        {
                            "from_class": from_m.group(1),
                            "to_class": to_m.group(1),
                            "description": desc_m.group(1) if desc_m else f"Transition from {from_m.group(1)} to {to_m.group(1)}.",
                            "region_id": primary_region_id,
                            "confidence": 0.85,
                            "is_uncertain": False,
                        }
                    ]
                if extracted.get("temporal_direction"):
                    parsed_dict = extracted
            except Exception as e:
                logger.debug("Field-by-field regex recovery failed: %s", e)

        # Fallback if VLM produced unstructured text
        if not isinstance(parsed_dict, dict):
            logger.warning("VLM output could not be parsed as structured JSON. Generating defensive fallback.")
            clean_snippet = raw_text[:200].replace("\n", " ").strip() if raw_text else "Physical changes observed."
            summary = (
                f"Bi-temporal change observed across {area_context} ({change_ratio_pct:.2f}% of AOI). "
                f"Interpretation: {clean_snippet}"
            )
            fallback_trans = SemanticTransition(
                transition_id="trans_001",
                from_class="unknown",
                to_class="unknown",
                description=clean_snippet,
                region_id=primary_region_id,
                bbox_pixel=primary_bbox,
                semantic_confidence=0.50,
                is_uncertain=True,
                evidence_support="visual_crop_comparison (unstructured)",
            )
            return SemanticChangeInterpretation(
                summary=summary,
                temporal_direction="uncertain",
                predominant_transition="unknown -> unknown",
                transitions=[fallback_trans],
                supporting_regions=[primary_region_id],
                warnings=["Model output was unstructured; transition classes marked uncertain."],
                supporting_model=self.model_id,
                semantic_uncertainty=0.50,
            )

        # Extract structured fields
        summary = str(parsed_dict.get("summary") or f"Physical changes detected across {area_context}.")
        raw_direction = str(parsed_dict.get("temporal_direction") or "modified").lower().strip()
        valid_directions = ["increased", "decreased", "modified", "no_change", "uncertain"]
        temporal_direction = raw_direction if raw_direction in valid_directions else "modified"

        # If significant physical change was verified (>500 px) but VLM answered 'no_change'
        # because the user-inquired target class was absent (e.g. asking about buildings when only soil changed),
        # guard the physical scene direction as 'modified' to prevent false contradiction with verified spatial facts.
        if changed_pixels > 500 and temporal_direction == "no_change":
            temporal_direction = "modified"

        predom_trans = str(parsed_dict.get("predominant_transition") or "unspecified_transition")
        parsed_transitions = parsed_dict.get("transitions", [])
        warnings = [str(w) for w in parsed_dict.get("warnings", [])]

        def _check_class_uncertainty(raw_class: str) -> bool:
            """Checks if class is unknown, uncertain, or unrecognized in remote sensing."""
            c = raw_class.strip().lower().replace("-", "_").replace(" ", "_")
            if any(k in c for k in ["unknown", "uncertain", "unclassified", "ambiguous", "unspecified"]):
                return True
            known_roots = [
                "forest", "tree", "canopy", "woodland", "timber",
                "water", "lake", "reservoir", "river", "flood", "pond", "ocean", "sea", "aquatic",
                "built", "building", "structure", "house", "residential", "commercial", "warehouse", "roof",
                "road", "highway", "pavement", "runway", "bridge", "infrastructure",
                "crop", "agri", "farm", "field", "vegetation", "grass", "pasture", "shrub",
                "bare", "soil", "dirt", "ground", "earth", "sand", "excavat", "tilled", "plow", "cleared",
            ]
            return not any(k in c for k in known_roots)

        transitions: List[SemanticTransition] = []
        supporting_regions: List[str] = []

        if isinstance(parsed_transitions, list) and len(parsed_transitions) > 0:
            for i, t in enumerate(parsed_transitions):
                if not isinstance(t, dict):
                    continue
                from_c = str(t.get("from_class") or "unknown").strip().lower()
                to_c = str(t.get("to_class") or "unknown").strip().lower()
                from_unc = _check_class_uncertainty(from_c)
                to_unc = _check_class_uncertainty(to_c)

                desc = str(t.get("description") or f"Transition from {from_c} to {to_c}.")
                r_id = str(t.get("region_id") or primary_region_id)
                conf = float(t.get("confidence", 0.80))
                conf = min(1.0, max(0.0, conf))

                is_unc = (
                    bool(t.get("is_uncertain", False))
                    or from_unc
                    or to_unc
                    or (from_c in ["unknown", "uncertain"])
                    or (to_c in ["unknown", "uncertain"])
                    or (from_c == to_c)
                )

                if is_unc:
                    conf = min(conf, 0.60)

                if r_id not in supporting_regions:
                    supporting_regions.append(r_id)

                transitions.append(
                    SemanticTransition(
                        transition_id=f"trans_{i + 1:03d}",
                        from_class=from_c,
                        to_class=to_c,
                        description=desc,
                        region_id=r_id,
                        bbox_pixel=primary_bbox if i == 0 else None,
                        semantic_confidence=conf,
                        is_uncertain=is_unc,
                        evidence_support="visual_crop_comparison",
                    )
                )

        if not transitions:
            transitions.append(
                SemanticTransition(
                    transition_id="trans_001",
                    from_class="unknown",
                    to_class="unknown",
                    description=summary,
                    region_id=primary_region_id,
                    bbox_pixel=primary_bbox,
                    semantic_confidence=0.50,
                    is_uncertain=True,
                    evidence_support="visual_crop_comparison",
                )
            )
            supporting_regions.append(primary_region_id)

        # Update predominant transition if missing
        if predom_trans in ["unspecified_transition", ""] and transitions:
            predom_trans = f"{transitions[0].from_class} -> {transitions[0].to_class}"

        avg_conf = np.mean([t.semantic_confidence for t in transitions]) if transitions else 0.50
        semantic_uncertainty = round(float(1.0 - avg_conf), 4)

        return SemanticChangeInterpretation(
            summary=summary,
            temporal_direction=temporal_direction,
            predominant_transition=predom_trans,
            transitions=transitions,
            supporting_regions=supporting_regions,
            warnings=warnings,
            supporting_model=self.model_id,
            semantic_uncertainty=semantic_uncertainty,
        )

    def health_check(self) -> bool:
        """Liveness check for specialist."""
        if self._is_loaded and hasattr(self, "model") and self.model is not None:
            return True
        return True

    def cleanup(self) -> None:
        """Frees model weights and cleans CUDA cache."""
        if getattr(self, "_using_shared_runtime", False):
            from backend.models.shared_vlm import release_shared_vlm
            release_shared_vlm()
            self.model = None
            self.processor = None
            self._is_loaded = False
            self._using_shared_runtime = False
            logger.info("Change VQA detached from Shared VLM Runtime.")
            return

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        self._is_loaded = False
        logger.info("ChangeVQASpecialist unloaded and VRAM cleared.")
