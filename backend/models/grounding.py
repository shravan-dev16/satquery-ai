"""Text-guided Remote-Sensing Region Grounding Specialists (Milestone M2.1).

Provides:
1. GroundingDinoSpecialist: Dedicated zero-shot grounding detector (IDEA-Research/grounding-dino-base).
2. QwenGroundingSpecialist: Baseline zero-shot multimodal VLM grounding (Qwen/Qwen2-VL-2B-Instruct).
3. RemoteSensingGroundingSpecialist: Dynamic specialist facade registered as RS_GROUND in ModelRegistry.
4. QueryNormalizer: Deterministic normalization of natural-language queries into detector entity phrases.
5. Box post-processing with Non-Maximum Suppression (NMS) and Degenerate Box Detection (>=98% coverage).

Strict Coordinate Convention:
- Origin (0, 0) at TOP-LEFT corner of raster
- x: horizontal column index (left-to-right)
- y: vertical row index (top-to-bottom)
- Ordering: strictly [xmin, ymin, xmax, ymax] everywhere
"""

import gc
import json
import logging
import re
import time
from pathlib import Path
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
    SpecialistInput,
    SpecialistOutput,
    TaskType,
)
from backend.evidence.spatial import PixelBoundingBox, SpatialTransformer
from backend.models.base import BaseSpecialist
from backend.preprocessing.geotiff import GeoTIFFReader

logger = logging.getLogger(__name__)


class QueryNormalizer:
    """Deterministically normalizes natural-language referring queries for detectors."""

    STRIP_PREFIXES = [
        "where is the", "where are the", "where is a", "where is an", "where are", "where is",
        "locate the", "locate a", "locate an", "locate",
        "find the", "find a", "find an", "find",
        "highlight the", "highlight a", "highlight an", "highlight",
        "detect the", "detect a", "detect an", "detect",
        "identify the", "identify a", "identify an", "identify",
        "show me the", "show me a", "show me",
        "can you find the", "can you locate the", "can you highlight the",
    ]

    @classmethod
    def normalize(cls, query: str) -> str:
        """Transforms user query into detector phrase with trailing period.

        Example:
            'Where is the water body?' -> 'water body.'
            'Locate the runway.' -> 'runway.'
            'Find the road.' -> 'road.'
        """
        cleaned = query.strip()
        cleaned_lower = cleaned.lower()

        # Strip interrogative/command prefix
        for prefix in sorted(cls.STRIP_PREFIXES, key=len, reverse=True):
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                cleaned_lower = cleaned.lower()
                break

        # Strip remaining leading articles
        for art in ["the ", "a ", "an "]:
            if cleaned_lower.startswith(art):
                cleaned = cleaned[len(art):].strip()
                cleaned_lower = cleaned.lower()
                break

        # Remove trailing punctuation
        cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()

        if not cleaned:
            cleaned = "object"

        # Grounding DINO BERT text encoder expects entity phrases ending with a period
        return f"{cleaned}."


def calculate_box_iou(
    box1: Union[Tuple[float, float, float, float], List[float]],
    box2: Union[Tuple[float, float, float, float], List[float]],
) -> float:
    """Computes Intersection over Union (IoU) between two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def apply_nms(boxes: List[PixelBoundingBox], iou_threshold: float = 0.70) -> List[PixelBoundingBox]:
    """Filters duplicate overlapping boxes keeping higher confidence predictions."""
    if not boxes:
        return []
    sorted_boxes = sorted(
        boxes,
        key=lambda b: (b.model_score if b.model_score is not None else (b.confidence or 0.0)),
        reverse=True,
    )
    kept: List[PixelBoundingBox] = []
    for b in sorted_boxes:
        suppress = False
        for k in kept:
            iou = calculate_box_iou((b.xmin, b.ymin, b.xmax, b.ymax), (k.xmin, k.ymin, k.xmax, k.ymax))
            if iou >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(b)
    return kept


class GroundingCoordinateParser:
    """Parses bounding box expressions from model output text into PixelBoundingBox objects."""

    @staticmethod
    def parse_boxes(
        text: str,
        image_width: int,
        image_height: int,
        target_label: str = "detected_region",
    ) -> List[PixelBoundingBox]:
        """Extracts bounding box coordinates from text output and maps to pixel coordinates."""
        parsed_boxes: List[PixelBoundingBox] = []

        # 1. Check for JSON structure
        try:
            json_matches = re.findall(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
            for jm in json_matches:
                try:
                    data = json.loads(jm)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict) and "box_2d" in item:
                            ymin, xmin, ymax, xmax = [float(v) for v in item["box_2d"]]
                            px_xmin = (xmin / 1000.0) * image_width
                            px_ymin = (ymin / 1000.0) * image_height
                            px_xmax = (xmax / 1000.0) * image_width
                            px_ymax = (ymax / 1000.0) * image_height
                            parsed_boxes.append(
                                PixelBoundingBox(
                                    xmin=px_xmin,
                                    ymin=px_ymin,
                                    xmax=px_xmax,
                                    ymax=px_ymax,
                                    label=item.get("label", target_label),
                                )
                            )
                except Exception:
                    pass
        except Exception:
            pass

        # 2. Check for Qwen2-VL <|box_start|>(ymin, xmin),(ymax, xmax)<|box_end|> tokens
        if not parsed_boxes:
            box_token_pat = re.findall(
                r"<\|box_start\|>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*<\|box_end\|>",
                text,
            )
            for ymin_s, xmin_s, ymax_s, xmax_s in box_token_pat:
                ymin, xmin, ymax, xmax = float(ymin_s), float(xmin_s), float(ymax_s), float(xmax_s)
                px_xmin = (xmin / 1000.0) * image_width
                px_ymin = (ymin / 1000.0) * image_height
                px_xmax = (xmax / 1000.0) * image_width
                px_ymax = (ymax / 1000.0) * image_height
                if px_xmin < px_xmax and px_ymin < px_ymax:
                    parsed_boxes.append(
                        PixelBoundingBox(
                            xmin=px_xmin,
                            ymin=px_ymin,
                            xmax=px_xmax,
                            ymax=px_ymax,
                            label=target_label,
                        )
                    )

        # 3. Check for coordinate bracket tags
        if not parsed_boxes:
            coord_pat = re.findall(
                r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",
                text,
            )
            for match in coord_pat:
                v1, v2, v3, v4 = [float(x) for x in match]
                if max(v1, v2, v3, v4) <= 1.0:
                    px_xmin, px_ymin, px_xmax, px_ymax = (
                        v1 * image_width,
                        v2 * image_height,
                        v3 * image_width,
                        v4 * image_height,
                    )
                elif max(v1, v2, v3, v4) <= 1000.0 and (
                    image_width > 1000
                    or image_height > 1000
                    or max(v1, v2, v3, v4) > min(image_width, image_height)
                ):
                    px_xmin = (v2 / 1000.0) * image_width
                    px_ymin = (v1 / 1000.0) * image_height
                    px_xmax = (v4 / 1000.0) * image_width
                    px_ymax = (v3 / 1000.0) * image_height
                else:
                    px_xmin, px_ymin, px_xmax, px_ymax = v1, v2, v3, v4

                if px_xmin < px_xmax and px_ymin < px_ymax:
                    parsed_boxes.append(
                        PixelBoundingBox(
                            xmin=px_xmin,
                            ymin=px_ymin,
                            xmax=px_xmax,
                            ymax=px_ymax,
                            label=target_label,
                        )
                    )

        # Clamping and filtering valid boxes
        valid_boxes: List[PixelBoundingBox] = []
        for box in parsed_boxes:
            clamped = box.clamp(image_width, image_height)
            if clamped.validate_bounds(image_width, image_height):
                valid_boxes.append(clamped)

        return valid_boxes


class GroundingDinoSpecialist(BaseSpecialist):
    """Dedicated Zero-Shot Grounding Specialist using IDEA-Research/grounding-dino-base."""

    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device=chosen_device)
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.processor = None
        self._load_duration_ms = 0

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier="RS_GROUND_DINO",
            name=f"Grounding DINO Remote-Sensing Grounding ({self.model_id})",
            version="1.0.0",
            task=TaskType.GROUNDING,
            supported_modalities=[ModalityType.OPTICAL, ModalityType.MULTISPECTRAL],
            supported_input_count=[1],
            requires_gpu=True,
            vram_budget_mb=3500,
            confidence_available=True,
            fallback_specialist_id="RS_GROUND_QWEN",
        )

    def load(self) -> None:
        if self._is_loaded:
            return

        start_time = time.perf_counter()
        logger.info("Loading Grounding DINO [%s] on [%s]...", self.model_id, self.device)

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)

        if self.device == "cuda":
            self.model = self.model.to("cuda")

        self.model.eval()
        self._is_loaded = True
        self._load_duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("Grounding DINO loaded in %d ms", self._load_duration_ms)

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        raw_query = inputs.query or "object"
        normalized_query = QueryNormalizer.normalize(raw_query)
        image_path = Path(inputs.primary_image_path)

        rgb_array, raw_meta = GeoTIFFReader.read_normalized_rgb(image_path)
        img_width = raw_meta["width"]
        img_height = raw_meta["height"]
        pil_image = Image.fromarray(rgb_array)

        # Run Grounding DINO detector
        inputs_tensor = self.processor(
            images=pil_image,
            text=normalized_query,
            return_tensors="pt",
        )
        if self.device == "cuda":
            inputs_tensor = inputs_tensor.to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs_tensor)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs_tensor.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(img_height, img_width)],
        )[0]

        boxes_tensor = results["boxes"]
        scores_tensor = results["scores"]
        labels_list = results.get("text_labels", results.get("labels", []))

        candidate_boxes: List[PixelBoundingBox] = []
        degenerate_boxes: List[PixelBoundingBox] = []

        for idx in range(len(scores_tensor)):
            b = boxes_tensor[idx].tolist()
            score = float(scores_tensor[idx].item())
            lbl = str(labels_list[idx]) if idx < len(labels_list) else raw_query

            pbox = PixelBoundingBox(
                xmin=b[0],
                ymin=b[1],
                xmax=b[2],
                ymax=b[3],
                label=lbl,
                confidence=round(score, 4),
                model_score=round(score, 4),
            )

            # Clamp and check boundaries
            clamped = pbox.clamp(img_width, img_height)
            if not clamped.validate_bounds(img_width, img_height):
                continue

            # Minimum area filter (16 px^2)
            box_area = (clamped.xmax - clamped.xmin) * (clamped.ymax - clamped.ymin)
            if box_area < 16.0:
                continue

            # Check degeneracy rule (>= 98% image coverage or full-image fallback)
            is_degen, reason = clamped.check_degeneracy(img_width, img_height, threshold_pct=0.98)
            if is_degen:
                clamped.is_degenerate = True
                clamped.degenerate_reason = reason
                degenerate_boxes.append(clamped)
            else:
                candidate_boxes.append(clamped)

        # Apply Non-Maximum Suppression (NMS) on non-degenerate candidates
        valid_boxes = apply_nms(candidate_boxes, iou_threshold=0.70)

        warnings: List[str] = []
        if not valid_boxes:
            if degenerate_boxes:
                warnings.append(
                    f"Detector produced {len(degenerate_boxes)} degenerate full-image candidate(s) which were excluded from valid results."
                )
            warnings.append(f"No valid grounded regions were detected for query '{raw_query}'.")

        # Project valid boxes to geospatial GeoJSON features
        affine_transform = raw_meta.get("transform")
        crs_str = raw_meta.get("crs")

        bounding_box_evidence: List[BoundingBox] = []
        detected_regions: List[DetectedRegion] = []

        for idx, pbox in enumerate(valid_boxes):
            norm_coords = pbox.to_normalized(img_width, img_height)

            geojson_geom = None
            if affine_transform:
                geo_region = SpatialTransformer.pixel_bbox_to_geospatial_polygon(
                    bbox=pbox,
                    affine_transform=affine_transform,
                    crs=crs_str,
                    width=img_width,
                    height=img_height,
                )
                geojson_geom = geo_region.geojson_feature["geometry"]

            box_item = BoundingBox(
                box_id=f"box_{idx + 1:03d}",
                label=pbox.label,
                confidence=pbox.confidence or 0.0,
                model_score=pbox.model_score,
                is_degenerate=False,
                coordinates_normalized=norm_coords,
                coordinates_pixel=(int(pbox.xmin), int(pbox.ymin), int(pbox.xmax), int(pbox.ymax)),
                geojson=geojson_geom,
            )
            bounding_box_evidence.append(box_item)

            detected_regions.append(
                DetectedRegion(
                    region_name=pbox.label,
                    label=pbox.label,
                    bbox_pixel=(pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax),
                    confidence=pbox.confidence,
                    model_score=pbox.model_score,
                    is_degenerate=False,
                    polygon_pixel=None,
                    details={
                        "bbox_pixel": [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax],
                        "model_score": pbox.model_score,
                        "coordinate_convention": "origin: top-left, x: right, y: down, order: [xmin, ymin, xmax, ymax]",
                        "is_axis_aligned_polygon": True,
                    },
                )
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Generate Web Preview Artifacts
        static_dir = Path("backend/static/previews")
        static_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"gnd_{int(time.time() * 1000) % 100000}"

        primary_filename = f"primary_preview_{run_id}.png"
        pil_image.save(static_dir / primary_filename, format="PNG")

        evidence_images = [
            EvidenceImage(
                role="primary",
                url=f"/api/v1/static/previews/{primary_filename}",
                width=img_width,
                height=img_height,
                crs=crs_str,
                bounds=raw_meta.get("bounds"),
            )
        ]

        if valid_boxes:
            preview_img = pil_image.copy()
            draw = ImageDraw.Draw(preview_img)
            for box in valid_boxes:
                draw.rectangle(
                    [box.xmin, box.ymin, box.xmax, box.ymax],
                    outline=(0, 255, 128),
                    width=3,
                )
                lbl_text = f"{box.label} ({box.confidence:.2f})" if box.confidence else box.label
                draw.text((box.xmin + 4, box.ymin + 4), lbl_text, fill=(0, 255, 128))

            grounding_filename = f"grounding_preview_{run_id}.png"
            preview_img.save(static_dir / grounding_filename, format="PNG")

            evidence_images.append(
                EvidenceImage(
                    role="grounding_preview",
                    url=f"/api/v1/static/previews/{grounding_filename}",
                    width=img_width,
                    height=img_height,
                    crs=crs_str,
                    bounds=raw_meta.get("bounds"),
                )
            )

        evidence = EvidenceBundle(
            images=evidence_images,
            masks=[],
            boxes=bounding_box_evidence,
            statistics=[],
            regions=detected_regions,
        )

        top_score = valid_boxes[0].model_score if valid_boxes else 0.0
        answer_summary = (
            f"Detected {len(valid_boxes)} grounded region(s) for '{raw_query}' (normalized: '{normalized_query}')."
            if valid_boxes
            else f"No valid grounded regions were detected for query '{raw_query}'."
        )

        return SpecialistOutput(
            specialist_id=self.capability.identifier,
            success=True,
            answer_text=answer_summary,
            confidence=top_score if valid_boxes else 0.0,
            evidence=evidence,
            parameters_used={
                "model_id": self.model_id,
                "original_query": raw_query,
                "normalized_query": normalized_query,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "coordinate_convention": "pixel [xmin, ymin, xmax, ymax]",
                "raw_candidate_count": len(scores_tensor),
                "degenerate_candidate_count": len(degenerate_boxes),
            },
            warnings=warnings,
            execution_time_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        if self._is_loaded:
            return self.model is not None and self.processor is not None
        return True

    def cleanup(self) -> None:
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
        logger.info("Grounding DINO specialist memory released.")


class QwenGroundingSpecialist(BaseSpecialist):
    """Baseline Zero-Shot Multimodal VLM Grounding using Qwen2-VL-2B-Instruct."""

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
        return ModelCapability(
            identifier="RS_GROUND_QWEN",
            name=f"Qwen2-VL Baseline Grounding ({self.model_id})",
            version="1.0.0",
            task=TaskType.GROUNDING,
            supported_modalities=[ModalityType.OPTICAL, ModalityType.MULTISPECTRAL],
            supported_input_count=[1],
            requires_gpu=True,
            vram_budget_mb=4500,
            confidence_available=True,
        )

    def load(self) -> None:
        if self._is_loaded:
            return

        start_time = time.perf_counter()
        logger.info("Loading Qwen Grounding baseline [%s] on [%s]...", self.model_id, self.device)

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

        self.model.eval()
        self._is_loaded = True
        self._load_duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("Qwen Grounding baseline loaded in %d ms", self._load_duration_ms)

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        if not self._is_loaded:
            self.load()

        start_time = time.perf_counter()
        referring_expr = inputs.query or "Locate the target feature."
        image_path = Path(inputs.primary_image_path)

        rgb_array, raw_meta = GeoTIFFReader.read_normalized_rgb(image_path)
        img_width = raw_meta["width"]
        img_height = raw_meta["height"]
        pil_image = Image.fromarray(rgb_array)

        grounding_prompt = (
            f"Locate and detect '{referring_expr}' in this overhead remote-sensing image. "
            f"Output the detected bounding box coordinates in JSON format: {{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"{referring_expr}\"}} "
            f"normalized to 0-1000 scale."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a remote-sensing vision grounding model. Detect geographic features and output bounding boxes.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": grounding_prompt},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_tensor = self.processor(
            text=[prompt_text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        if self.device == "cuda":
            inputs_tensor = inputs_tensor.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs_tensor,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs_tensor.input_ids, generated_ids.sequences)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        parsed_boxes = GroundingCoordinateParser.parse_boxes(
            text=output_text,
            image_width=img_width,
            image_height=img_height,
            target_label=referring_expr,
        )

        valid_boxes: List[PixelBoundingBox] = []
        degenerate_boxes: List[PixelBoundingBox] = []

        for pbox in parsed_boxes:
            is_degen, reason = pbox.check_degeneracy(img_width, img_height, threshold_pct=0.98)
            if is_degen:
                pbox.is_degenerate = True
                pbox.degenerate_reason = reason
                degenerate_boxes.append(pbox)
            else:
                valid_boxes.append(pbox)

        warnings: List[str] = []
        if not valid_boxes:
            if degenerate_boxes:
                warnings.append(
                    f"Qwen produced {len(degenerate_boxes)} degenerate full-image fallback box(es) which were excluded from valid results."
                )
            warnings.append(f"No valid grounded regions were detected for query '{referring_expr}'.")

        affine_transform = raw_meta.get("transform")
        crs_str = raw_meta.get("crs")

        bounding_box_evidence: List[BoundingBox] = []
        detected_regions: List[DetectedRegion] = []

        for idx, pbox in enumerate(valid_boxes):
            norm_coords = pbox.to_normalized(img_width, img_height)

            geojson_geom = None
            if affine_transform:
                geo_region = SpatialTransformer.pixel_bbox_to_geospatial_polygon(
                    bbox=pbox,
                    affine_transform=affine_transform,
                    crs=crs_str,
                    width=img_width,
                    height=img_height,
                )
                geojson_geom = geo_region.geojson_feature["geometry"]

            box_item = BoundingBox(
                box_id=f"box_{idx + 1:03d}",
                label=pbox.label,
                confidence=0.50,
                model_score=None,
                is_degenerate=False,
                coordinates_normalized=norm_coords,
                coordinates_pixel=(int(pbox.xmin), int(pbox.ymin), int(pbox.xmax), int(pbox.ymax)),
                geojson=geojson_geom,
            )
            bounding_box_evidence.append(box_item)

            detected_regions.append(
                DetectedRegion(
                    region_name=pbox.label,
                    label=pbox.label,
                    bbox_pixel=(pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax),
                    confidence=0.50,
                    is_degenerate=False,
                    polygon_pixel=None,
                    details={
                        "bbox_pixel": [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax],
                        "coordinate_convention": "pixel [xmin, ymin, xmax, ymax]",
                        "is_axis_aligned_polygon": True,
                    },
                )
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Generate Web Preview Artifacts
        static_dir = Path("backend/static/previews")
        static_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"qgnd_{int(time.time() * 1000) % 100000}"

        primary_filename = f"primary_preview_{run_id}.png"
        pil_image.save(static_dir / primary_filename, format="PNG")

        evidence_images = [
            EvidenceImage(
                role="primary",
                url=f"/api/v1/static/previews/{primary_filename}",
                width=img_width,
                height=img_height,
                crs=crs_str,
                bounds=raw_meta.get("bounds"),
            )
        ]

        if valid_boxes:
            preview_img = pil_image.copy()
            draw = ImageDraw.Draw(preview_img)
            for box in valid_boxes:
                draw.rectangle(
                    [box.xmin, box.ymin, box.xmax, box.ymax],
                    outline=(0, 255, 128),
                    width=3,
                )
                lbl_text = f"{box.label} ({box.confidence:.2f})" if box.confidence else box.label
                draw.text((box.xmin + 4, box.ymin + 4), lbl_text, fill=(0, 255, 128))

            grounding_filename = f"grounding_preview_{run_id}.png"
            preview_img.save(static_dir / grounding_filename, format="PNG")

            evidence_images.append(
                EvidenceImage(
                    role="grounding_preview",
                    url=f"/api/v1/static/previews/{grounding_filename}",
                    width=img_width,
                    height=img_height,
                    crs=crs_str,
                    bounds=raw_meta.get("bounds"),
                )
            )

        evidence = EvidenceBundle(
            images=evidence_images,
            masks=[],
            boxes=bounding_box_evidence,
            statistics=[],
            regions=detected_regions,
        )

        return SpecialistOutput(
            specialist_id=self.capability.identifier,
            success=True,
            answer_text=(
                f"Detected {len(valid_boxes)} region(s) matching '{referring_expr}'."
                if valid_boxes
                else f"No valid grounded regions were detected for query '{referring_expr}'."
            ),
            confidence=0.50 if valid_boxes else 0.0,
            evidence=evidence,
            parameters_used={
                "model_id": self.model_id,
                "original_query": referring_expr,
                "raw_candidate_count": len(parsed_boxes),
                "degenerate_candidate_count": len(degenerate_boxes),
            },
            warnings=warnings,
            execution_time_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        if self._is_loaded:
            return self.model is not None and self.processor is not None
        return True

    def cleanup(self) -> None:
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
        logger.info("Qwen Grounding baseline memory released.")


class RemoteSensingGroundingSpecialist(BaseSpecialist):
    """Facade for the RS_GROUND capability in ModelRegistry.

    By default, dispatches to GroundingDinoSpecialist (IDEA-Research/grounding-dino-base),
    while permitting switching to QwenGroundingSpecialist for evaluation.
    """

    def __init__(
        self,
        candidate: str = "grounding_dino",
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device=chosen_device)
        self.candidate = candidate
        if candidate == "qwen":
            self.active_specialist = QwenGroundingSpecialist(model_id=model_id, device=chosen_device)
        else:
            self.active_specialist = GroundingDinoSpecialist(model_id=model_id, device=chosen_device)

    @property
    def capability(self) -> ModelCapability:
        cap = self.active_specialist.capability
        # Ensure registry identifier remains RS_GROUND
        return ModelCapability(
            identifier="RS_GROUND",
            name=f"Remote-Sensing Region Grounding ({cap.name})",
            version="2.1.0",
            task=TaskType.GROUNDING,
            supported_modalities=cap.supported_modalities,
            supported_input_count=cap.supported_input_count,
            requires_gpu=cap.requires_gpu,
            vram_budget_mb=cap.vram_budget_mb,
            confidence_available=cap.confidence_available,
            fallback_specialist_id=cap.fallback_specialist_id,
        )

    def load(self) -> None:
        self.active_specialist.load()
        self._is_loaded = self.active_specialist.is_loaded

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        output = self.active_specialist.predict(inputs)
        # Ensure specialist_id in output matches RS_GROUND for API routing
        output.specialist_id = "RS_GROUND"
        return output

    def health_check(self) -> bool:
        return self.active_specialist.health_check()

    def cleanup(self) -> None:
        self.active_specialist.cleanup()
        self._is_loaded = False
