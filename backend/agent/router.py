"""Agentic Query and Input Router (Milestone M7).

Adheres to SIH26167 requirement:
"The system must automatically select, sequence, and execute the appropriate specialist
models or tools according to the query and input configuration."

Inspects natural-language query semantics, input raster cardinality, and sensor modalities
to route queries dynamically to specialist tools, rejecting impossible combinations.
"""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.agent.schema import ModalityType, RoutingDecision, TaskType
from backend.preprocessing.modality import ModalityDetector

logger = logging.getLogger(__name__)


class AgentRouter:
    """Deterministic, auditable query and input configuration router."""

    # -----------------------------------------------------------------------
    # Query Intent Pattern Banks
    # -----------------------------------------------------------------------
    CROSS_MODAL_PATTERNS = [
        r"\boptical\s+(?:and|\+)\s+sar\b",
        r"\bsar\s+(?:and|\+)\s+optical\b",
        r"\bboth\s+sensors\b",
        r"\bcross[- ]modal\b",
        r"\bjoint\s+analysis\b",
        r"\busing\s+both\s+sensors\b",
        r"\bwith\s+sar\b",
        r"\bsar\s+backscatter\b",
        r"\boptical\s+and\s+radar\b",
        r"\bradar\s+and\s+optical\b",
        r"\btogether\s+to\s+identify\b",
        r"\bdual[- ]sensor\b",
    ]

    TEMPORAL_CHANGE_PATTERNS = [
        r"\bwhat\s+changed\b",
        r"\bbetween\s+these\s+(?:two\s+)?dates\b",
        r"\bbetween\s+(?:the\s+)?two\s+images\b",
        r"\btemporal\s+change\b",
        r"\bbi[- ]temporal\b",
        r"\bchanged\s+regions\b",
        r"\bdifference\s+between\b",
        r"\bover\s+time\b",
        r"\bchange\s+detection\b",
        r"\bhas\s+the\b.*\bchanged\b",
    ]

    # Pure total/binary change queries (explicitly requesting generic surface change without semantic target)
    PURE_TOTAL_CHANGE_PATTERNS = [
        r"^\s*what\s+changed(\s+between\s+(these\s+(two\s+)?dates|the\s+two\s+images))?\s*\??\s*$",
        r"\bhow\s+much\s+(?:total\s+)?area\s+changed\b",
        r"\bhow\s+much\s+(?:total\s+)?change\b",
        r"^\s*is\s+there\s+any\s+change\s*\??\s*$",
        r"^\s*has\s+the\s+scene\s+changed\s*\??\s*$",
        r"\btotal\s+(?:surface\s+|physical\s+)?change\b",
        r"\btotal\s+(?:changed\s+)?area\b",
        r"\boverall\s+change\b",
        r"\bbinary\s+change\b",
        r"^\s*(?:detect|show|find|highlight)\s+(?:the\s+)?changes?\s*\??\s*$",
    ]

    # Specific semantic classes from the remote sensing taxonomy
    SEMANTIC_CLASS_PATTERNS = [
        r"\b(?:buildings?|structures?|houses?|settlements?|residential|commercial|industrial|urban|built[- ]up)\b",
        r"\b(?:vegetation|crops?|cropland|farmland|pasture|canopy|grassland|agricultural)\b",
        r"\b(?:forests?|trees?|woodlands?|deforestation|timber)\b",
        r"\b(?:water(?:[- ]body)?|lakes?|rivers?|reservoirs?|ponds?|floods?|flooding|wetlands?)\b",
        r"\b(?:bare[- ]ground|bare[- ]soil|soils?|dirt|sand|cleared[- ]land|unpaved)\b",
        r"\b(?:roads?|highways?|runways?|bridges?|infrastructure|pavement)\b",
    ]

    # Semantic change verbs, transitions, directions, and qualitative expressions
    SEMANTIC_CHANGE_PATTERNS = [
        r"\bdescribe\b",
        r"\bexplain\b",
        r"\bhow\s+did\b",
        r"\bwhy\s+did\b",
        r"\bwhat\s+does\b",
        r"\btype\s+of\s+land[- ]cover\b",
        r"\btype\s+of\s+change\b",
        r"\bkind\s+of\s+change\b",
        r"\bnature\s+of\s+change\b",
        r"\bland[- ]cover\s+change\b",
        r"\bland[- ]use\s+change\b",
        r"\bincreased?\b",
        r"\bdecreased?\b",
        r"\bexpanded?\b",
        r"\bshrink(?:ing|s)?\b",
        r"\bshrunk\b",
        r"\breduced?\b",
        r"\blost\b",
        r"\bloss\b",
        r"\bcleared?\b",
        r"\bclearance\b",
        r"\bnewly\s+created\b",
        r"\bcreated\b",
        r"\bdeveloped?\b",
        r"\bdevelopment\b",
        r"\bconstructed?\b",
        r"\bconstruction\b",
        r"\bappeared\b",
        r"\bdemolished?\b",
        r"\bconverted?\b",
        r"\btransformed?\b",
        r"\bsignificant\s+change\b",
        r"\bwhere\s+did\b",
        r"\bwhere\s+were\b",
        r"\bsemantic\b",
        r"\bmeaning\b",
        r"\binterpretation\b",
    ]

    GROUNDING_PATTERNS = [
        r"^\s*where\s+is\b",
        r"\blocate\b",
        r"\bground\b",
        r"\bfind\s+the\b",
        r"\bhighlight\b",
        r"\bbounding\s+box\b",
        r"\bdetect\s+the\b",
        r"\bshow\s+me\s+where\b",
        r"\bsegment\b",
        r"\bsegmentation\b",
    ]

    CAPTION_PATTERNS = [
        r"\bdescribe\s+the\s+scene\b",
        r"\bgenerate\s+(?:a\s+)?caption\b",
        r"\bscene\s+description\b",
        r"\bcaption\s+this\b",
        r"\bsummarize\s+(?:the\s+)?scene\b",
        r"\boverview\s+of\s+(?:the\s+)?image\b",
    ]

    @classmethod
    def route(
        cls,
        query: str,
        primary_path: Union[str, Path],
        secondary_path: Optional[Union[str, Path]] = None,
        task_hint: Optional[str] = None,
        primary_filename: Optional[str] = None,
        secondary_filename: Optional[str] = None,
    ) -> RoutingDecision:
        """Determines analytical route and specialist sequence from query and inputs."""
        q_clean = (query or "").strip()
        q_lower = q_clean.lower()
        p1 = Path(primary_path)
        p2 = Path(secondary_path) if secondary_path else None

        has_secondary = p2 is not None

        # -------------------------------------------------------------------
        # 1. Modality Inspection of Inputs
        # -------------------------------------------------------------------
        mod1 = ModalityDetector.identify(p1).modality
        mod2 = ModalityDetector.identify(p2).modality if p2 else None

        is_m1_optical = mod1 in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL)
        is_m1_sar = mod1 == ModalityType.SAR

        is_m2_optical = mod2 in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL) if mod2 else False
        is_m2_sar = mod2 == ModalityType.SAR if mod2 else False

        is_cross_modal_pair = has_secondary and (
            (is_m1_optical and is_m2_sar) or (is_m1_sar and is_m2_optical)
        )
        is_dual_optical = has_secondary and (is_m1_optical and is_m2_optical)
        is_dual_sar = has_secondary and (is_m1_sar and is_m2_sar)

        if not has_secondary:
            input_config = f"single_{mod1.value}"
        elif is_cross_modal_pair:
            input_config = "cross_modal_optical_sar"
        elif is_dual_optical:
            input_config = "bitemporal_optical_pair"
        elif is_dual_sar:
            input_config = "bitemporal_sar_pair"
        else:
            input_config = f"dual_{mod1.value}_{mod2.value}"

        # -------------------------------------------------------------------
        # 2. Intent Analysis
        # -------------------------------------------------------------------
        is_cross_modal_intent = any(re.search(pat, q_lower) for pat in cls.CROSS_MODAL_PATTERNS)
        is_temporal_intent = any(re.search(pat, q_lower) for pat in cls.TEMPORAL_CHANGE_PATTERNS)

        has_semantic_class = any(re.search(pat, q_lower) for pat in cls.SEMANTIC_CLASS_PATTERNS)
        has_semantic_action = any(re.search(pat, q_lower) for pat in cls.SEMANTIC_CHANGE_PATTERNS)
        is_pure_total_change = any(re.search(pat, q_lower) for pat in cls.PURE_TOTAL_CHANGE_PATTERNS)

        # Distinguish pure total change queries from semantic change queries:
        # A query is semantic if it references a semantic class (e.g. buildings, vegetation)
        # OR requests qualitative/semantic interpretation ("type of land-cover", "increase/decrease", "loss")
        # unless it is an explicitly pure total change query with zero semantic target.
        if has_semantic_class:
            is_semantic_change_intent = True
        elif is_pure_total_change:
            is_semantic_change_intent = False
        else:
            is_semantic_change_intent = has_semantic_action

        is_grounding_intent = any(re.search(pat, q_lower) for pat in cls.GROUNDING_PATTERNS)
        is_caption_intent = any(re.search(pat, q_lower) for pat in cls.CAPTION_PATTERNS)

        # -------------------------------------------------------------------
        # 3. Manual Task Hint Handling (Debug Override)
        # -------------------------------------------------------------------
        if task_hint and task_hint.lower().strip() not in ("", "auto", "none"):
            hint = task_hint.lower().strip()
            if "optical_sar" in hint or "fusion" in hint or "cross_modal" in hint:
                if not has_secondary:
                    return RoutingDecision(
                        task=TaskType.OPTICAL_SAR_ANALYSIS,
                        target_specialist_ids=[],
                        intent_category="cross_modal_fusion",
                        input_configuration=input_config,
                        is_valid=False,
                        rejection_reason="Cross-modal Optical-SAR joint analysis requires two images (one Optical and one SAR), but only a single image was provided.",
                        reasoning="Manual hint requested Optical-SAR analysis, but secondary image is missing.",
                    )
                if not is_cross_modal_pair:
                    err = (
                        "Cross-modal Optical-SAR joint analysis requires exactly one Optical image and one SAR image. "
                        f"Detected configuration: {input_config}."
                    )
                    return RoutingDecision(
                        task=TaskType.OPTICAL_SAR_ANALYSIS,
                        target_specialist_ids=[],
                        intent_category="cross_modal_fusion",
                        input_configuration=input_config,
                        is_valid=False,
                        rejection_reason=err,
                        reasoning="Manual hint requested Optical-SAR analysis, but modalities do not form an Optical + SAR pair.",
                    )
                return RoutingDecision(
                    task=TaskType.OPTICAL_SAR_ANALYSIS,
                    target_specialist_ids=["OPTICAL_SAR_FUSION"],
                    intent_category="cross_modal_fusion",
                    input_configuration=input_config,
                    is_multi_stage=False,
                    reasoning="Task hint explicitly requested Optical-SAR analysis. Validated cross-modal pair.",
                )

            if "change_vqa" in hint or "semantic" in hint:
                if not has_secondary:
                    return RoutingDecision(
                        task=TaskType.CHANGE_VQA,
                        target_specialist_ids=[],
                        intent_category="semantic_change_vqa",
                        input_configuration=input_config,
                        is_valid=False,
                        rejection_reason="Bi-temporal change analysis requires both primary (T1) and secondary (T2) images, but only a single image was provided.",
                        reasoning="Manual hint requested change VQA, but secondary image is missing.",
                    )
                return RoutingDecision(
                    task=TaskType.CHANGE_VQA,
                    target_specialist_ids=["CHANGE_DETECT", "CHANGE_VQA"],
                    intent_category="semantic_change_vqa",
                    input_configuration=input_config,
                    is_multi_stage=True,
                    reasoning="Task hint requested semantic change VQA. Sequenced TinyCD detection followed by Change VQA.",
                )

            if "change_detection" in hint or hint in ("tinycd_raw", "cva_raw"):
                if not has_secondary:
                    return RoutingDecision(
                        task=TaskType.CHANGE_DETECTION,
                        target_specialist_ids=[],
                        intent_category="binary_change_detection",
                        input_configuration=input_config,
                        is_valid=False,
                        rejection_reason="Bi-temporal change detection requires both primary (T1) and secondary (T2) images, but only a single image was provided.",
                        reasoning="Manual hint requested change detection, but secondary image is missing.",
                    )
                return RoutingDecision(
                    task=TaskType.CHANGE_DETECTION,
                    target_specialist_ids=["CHANGE_DETECT"],
                    intent_category="binary_change_detection",
                    input_configuration=input_config,
                    is_multi_stage=False,
                    reasoning="Task hint requested raw binary change detection without semantic interpretation.",
                )

            if "ground" in hint:
                return RoutingDecision(
                    task=TaskType.GROUNDING,
                    target_specialist_ids=["RS_GROUND"],
                    intent_category="region_grounding",
                    input_configuration=input_config,
                    reasoning="Task hint requested spatial region grounding.",
                )

            if "caption" in hint:
                return RoutingDecision(
                    task=TaskType.CAPTION,
                    target_specialist_ids=["RS_CAPTION"],
                    intent_category="scene_captioning",
                    input_configuration=input_config,
                    reasoning="Task hint requested scene captioning.",
                )

            if "vqa" in hint and not has_secondary:
                return RoutingDecision(
                    task=TaskType.VQA,
                    target_specialist_ids=["RS_VQA"],
                    intent_category="single_image_vqa",
                    input_configuration=input_config,
                    reasoning="Task hint requested single-image VQA.",
                )

        # -------------------------------------------------------------------
        # 4. Incompatible Combination Validation (Anti-Reinterpretation)
        # -------------------------------------------------------------------
        # Rule A: Temporal question on single image
        if not has_secondary and (is_temporal_intent or ("change" in q_lower and "image" in q_lower)):
            return RoutingDecision(
                task=TaskType.CHANGE_DETECTION,
                target_specialist_ids=[],
                intent_category="temporal_change",
                input_configuration=input_config,
                is_valid=False,
                rejection_reason="Temporal change analysis requires both Time 1 (earlier) and Time 2 (later) observations, but only a single image was uploaded.",
                reasoning="Detected temporal change query, but single image input configuration cannot satisfy bi-temporal analysis.",
            )

        # Rule B: Cross-modal query on single image
        if not has_secondary and is_cross_modal_intent:
            return RoutingDecision(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                target_specialist_ids=[],
                intent_category="cross_modal_fusion",
                input_configuration=input_config,
                is_valid=False,
                rejection_reason="Cross-modal optical-SAR analysis requires two images (one Optical and one SAR), but only a single image was uploaded.",
                reasoning="Detected cross-modal query, but only one image was provided.",
            )

        # Rule C: Cross-modal query on two optical images
        if is_dual_optical and is_cross_modal_intent:
            return RoutingDecision(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                target_specialist_ids=[],
                intent_category="cross_modal_fusion",
                input_configuration=input_config,
                is_valid=False,
                rejection_reason="Cross-modal optical-SAR joint analysis requires exactly one Optical image and one SAR image, but both uploaded images are Optical/Multispectral.",
                reasoning="Query requested cross-modal optical and SAR fusion, but both inputs are optical.",
            )

        # Rule D: Cross-modal query on two SAR images
        if is_dual_sar and is_cross_modal_intent:
            return RoutingDecision(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                target_specialist_ids=[],
                intent_category="cross_modal_fusion",
                input_configuration=input_config,
                is_valid=False,
                rejection_reason="Cross-modal optical-SAR joint analysis requires exactly one Optical image and one SAR image, but both uploaded images are SAR.",
                reasoning="Query requested cross-modal optical and SAR fusion, but both inputs are SAR.",
            )

        # Rule E: Pure temporal query on Optical + SAR pair (physical sensor mismatch)
        if is_cross_modal_pair and (is_temporal_intent or "what changed" in q_lower) and not is_cross_modal_intent:
            return RoutingDecision(
                task=TaskType.CHANGE_DETECTION,
                target_specialist_ids=[],
                intent_category="temporal_change",
                input_configuration=input_config,
                is_valid=False,
                rejection_reason=(
                    "Incompatible input configuration: Temporal change detection requires same-sensor observations "
                    "to prevent false change alarms caused by sensor physics mismatch. Uploaded imagery consists of "
                    "one Optical image and one SAR image. For this pair, use a cross-modal joint land-cover query."
                ),
                reasoning="Uploaded pair is Optical + SAR, but query is a pure bi-temporal change query.",
            )

        # -------------------------------------------------------------------
        # 5. Route Resolution
        # -------------------------------------------------------------------
        # Case A: Cross-Modal Optical + SAR Joint Analysis
        if is_cross_modal_pair:
            return RoutingDecision(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                target_specialist_ids=["OPTICAL_SAR_FUSION"],
                intent_category="cross_modal_fusion",
                input_configuration=input_config,
                is_multi_stage=False,
                reasoning="Co-registered Optical and SAR pair detected. Routed to OPTICAL_SAR_FUSION for joint raster feature fusion.",
            )

        # Case B: Bi-Temporal Imagery (Two same-modality rasters)
        if has_secondary:
            if is_semantic_change_intent:
                return RoutingDecision(
                    task=TaskType.CHANGE_VQA,
                    target_specialist_ids=["CHANGE_DETECT", "CHANGE_VQA"],
                    intent_category="semantic_change_vqa",
                    input_configuration=input_config,
                    is_multi_stage=True,
                    reasoning=(
                        "Bi-temporal raster pair with semantic query detected. Sequenced CHANGE_DETECT (TinyCD) "
                        "for spatial localization and extent, followed by CHANGE_VQA for grounded semantic land-cover transition interpretation."
                    ),
                )
            return RoutingDecision(
                task=TaskType.CHANGE_DETECTION,
                target_specialist_ids=["CHANGE_DETECT"],
                intent_category="bitemporal_change",
                input_configuration=input_config,
                is_multi_stage=False,
                reasoning=(
                    "Bi-temporal raster pair with binary change detection query detected. Routed to CHANGE_DETECT (TinyCD) "
                    "for spatial change mask and cluster localization."
                ),
            )

        # Case C: Single-Image Analysis
        if is_grounding_intent:
            return RoutingDecision(
                task=TaskType.GROUNDING,
                target_specialist_ids=["RS_GROUND"],
                intent_category="region_grounding",
                input_configuration=input_config,
                is_multi_stage=False,
                reasoning="Query semantics express spatial localization intent ('where is', 'locate', 'ground'). Routed to RS_GROUND.",
            )

        if is_caption_intent:
            return RoutingDecision(
                task=TaskType.CAPTION,
                target_specialist_ids=["RS_CAPTION"],
                intent_category="scene_captioning",
                input_configuration=input_config,
                is_multi_stage=False,
                reasoning="Query semantics express scene description intent ('describe scene', 'generate caption'). Routed to RS_CAPTION.",
            )

        # Default Single-Image VQA
        return RoutingDecision(
            task=TaskType.VQA,
            target_specialist_ids=["RS_VQA"],
            intent_category="single_image_vqa",
            input_configuration=input_config,
            is_multi_stage=False,
            reasoning="Single raster with analytical question. Routed to RS_VQA specialist.",
        )
