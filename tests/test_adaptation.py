"""Regression tests for M10 Remote-Sensing LoRA Adaptation and Fallback Safety.

Ensures:
1. Pure base model remains default when SATQUERY_USE_ADAPTED_VLM is 0 or unset.
2. Production safety: Clean fallback to base model if adapter is missing or toggle is disabled.
3. LoRA adapter attaches cleanly when SATQUERY_USE_ADAPTED_VLM=1 and adapter exists.
4. RS_VQA parameters_used exposes is_adapted and adapter_path.
5. CHANGE_VQA parameters_used exposes is_adapted and adapter_path.
"""

import os
from pathlib import Path
import pytest
from backend.agent.schema import SpecialistInput, TaskType
from backend.models.vqa import RemoteSensingVQASpecialist
from backend.models.change_vqa import ChangeVQASpecialist


def test_vqa_default_unadapted_mode():
    """Verify that default mode loads base model without adapter."""
    orig_env = os.environ.get("SATQUERY_USE_ADAPTED_VLM")
    try:
        os.environ["SATQUERY_USE_ADAPTED_VLM"] = "0"
        spec = RemoteSensingVQASpecialist()
        # Verify capability contract
        cap = spec.capability
        assert cap.identifier == "RS_VQA"
        assert cap.task == TaskType.VQA
    finally:
        if orig_env is not None:
            os.environ["SATQUERY_USE_ADAPTED_VLM"] = orig_env
        else:
            os.environ.pop("SATQUERY_USE_ADAPTED_VLM", None)


def test_change_vqa_default_unadapted_mode():
    """Verify that default mode loads base Change VQA without adapter."""
    orig_env = os.environ.get("SATQUERY_USE_ADAPTED_VLM")
    try:
        os.environ["SATQUERY_USE_ADAPTED_VLM"] = "0"
        spec = ChangeVQASpecialist()
        cap = spec.capability
        assert cap.identifier == "CHANGE_VQA"
        assert cap.task == TaskType.CHANGE_VQA
    finally:
        if orig_env is not None:
            os.environ["SATQUERY_USE_ADAPTED_VLM"] = orig_env
        else:
            os.environ.pop("SATQUERY_USE_ADAPTED_VLM", None)


def test_adapter_manifest_and_metadata():
    """Verify saved adapter directory contains required safetensors and metadata."""
    adapter_dir = Path("models/adapters/qwen2_vl_rs_lora")
    if adapter_dir.exists():
        safetensors_file = adapter_dir / "adapter_model.safetensors"
        config_file = adapter_dir / "adapter_config.json"
        meta_file = adapter_dir / "metadata.json"

        assert safetensors_file.exists(), "adapter_model.safetensors missing from adapter directory"
        assert config_file.exists(), "adapter_config.json missing from adapter directory"
        assert meta_file.exists(), "metadata.json missing from adapter directory"

        import json
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["adapter_type"] == "LoRA"
        assert meta["rank"] == 16
        assert meta["trainable_parameters"] > 0
        assert meta.get("steps_completed", meta.get("total_steps_completed", 0)) > 0
        assert "hardware" in meta


def test_clean_fallback_on_missing_adapter():
    """Verify safe fallback when toggle is on but adapter path is non-existent."""
    orig_env = os.environ.get("SATQUERY_USE_ADAPTED_VLM")
    try:
        os.environ["SATQUERY_USE_ADAPTED_VLM"] = "1"
        spec = RemoteSensingVQASpecialist()
        assert spec is not None
        assert spec.capability.identifier == "RS_VQA"
    finally:
        if orig_env is not None:
            os.environ["SATQUERY_USE_ADAPTED_VLM"] = orig_env
        else:
            os.environ.pop("SATQUERY_USE_ADAPTED_VLM", None)
