"""Unit & Concurrency Tests for Milestone M10.5 Shared VLM Runtime.

Verifies:
1. Singleton identity & ref counting
2. Model sharing between RS_VQA and CHANGE_VQA
3. Thread safety & concurrency safety
4. Clean memory release
5. Backward-compatibility toggle (SATQUERY_SHARED_VLM_RUNTIME=0)
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock, patch

from backend.models.shared_vlm import (
    SharedVLMRuntime,
    cleanup_shared_vlm,
    get_shared_vlm,
    release_shared_vlm,
)
from backend.models.vqa import RemoteSensingVQASpecialist
from backend.models.change_vqa import ChangeVQASpecialist


def test_shared_vlm_singleton_identity_mocked():
    """Verifies that SharedVLMRuntime reuses the exact same handle across acquisitions."""
    runtime = SharedVLMRuntime()
    with patch("transformers.AutoProcessor.from_pretrained") as mock_proc, \
         patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model:
        mock_proc.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        h1 = runtime.acquire(model_id="mock/model", device="cpu")
        assert runtime._ref_count == 1
        assert h1.model_id == "mock/model"

        # Second acquisition with same config should return same handle without reloading
        h2 = runtime.acquire(model_id="mock/model", device="cpu")
        assert runtime._ref_count == 2
        assert h1 is h2
        assert h1.model is h2.model

        # Release once
        runtime.release()
        assert runtime._ref_count == 1
        assert runtime.is_loaded

        # Release second time triggers cleanup
        runtime.release()
        assert runtime._ref_count == 0
        assert not runtime.is_loaded


def test_vqa_and_change_vqa_share_instance():
    """Verifies that RS_VQA and CHANGE_VQA attach to the exact same shared model instance."""
    cleanup_shared_vlm()
    os.environ["SATQUERY_SHARED_VLM_RUNTIME"] = "1"

    with patch("transformers.AutoProcessor.from_pretrained") as mock_proc, \
         patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model:
        fake_model = MagicMock()
        fake_proc = MagicMock()
        mock_model.return_value = fake_model
        mock_proc.return_value = fake_proc

        vqa = RemoteSensingVQASpecialist(device="cpu")
        change_vqa = ChangeVQASpecialist(device="cpu")

        vqa.load()
        change_vqa.load()

        assert vqa.is_loaded
        assert change_vqa.is_loaded
        # Identity check: they MUST share the same underlying model and processor instances
        assert vqa.model is change_vqa.model
        assert vqa.processor is change_vqa.processor

        # Check cleanup behavior
        vqa.cleanup()
        assert not vqa.is_loaded
        # Shared runtime should still be alive because change_vqa has a reference
        assert SharedVLMRuntime.get_instance().is_loaded

        change_vqa.cleanup()
        assert not change_vqa.is_loaded
        assert not SharedVLMRuntime.get_instance().is_loaded


def test_concurrency_thread_safety():
    """Verifies that concurrent acquisitions from multiple worker threads are thread-safe."""
    cleanup_shared_vlm()

    with patch("transformers.AutoProcessor.from_pretrained") as mock_proc, \
         patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        mock_proc.return_value = MagicMock()

        def worker_task(thread_id: int):
            handle = get_shared_vlm(model_id="Qwen/test-vlm", device="cpu")
            assert handle is not None
            # Simulate inference call
            with SharedVLMRuntime.get_instance().inference_lock:
                _ = f"result_from_thread_{thread_id}"
            release_shared_vlm()
            return True

        # Run 8 concurrent worker threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(8)]
            results = [f.result() for f in futures]

        assert all(results)
        cleanup_shared_vlm()


def test_reversibility_toggle():
    """Verifies that SATQUERY_SHARED_VLM_RUNTIME=0 bypasses the shared runtime cleanly."""
    cleanup_shared_vlm()
    os.environ["SATQUERY_SHARED_VLM_RUNTIME"] = "0"

    try:
        with patch("transformers.AutoProcessor.from_pretrained") as mock_proc, \
             patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained") as mock_model:
            mock_model.side_effect = [MagicMock(), MagicMock()]
            mock_proc.side_effect = [MagicMock(), MagicMock()]

            vqa = RemoteSensingVQASpecialist(device="cpu")
            change_vqa = ChangeVQASpecialist(device="cpu")

            vqa.load()
            change_vqa.load()

            # Under SATQUERY_SHARED_VLM_RUNTIME=0, they must have separate independent instances
            assert vqa.model is not change_vqa.model
            assert not getattr(vqa, "_using_shared_runtime", False)
            assert not getattr(change_vqa, "_using_shared_runtime", False)

            vqa.cleanup()
            change_vqa.cleanup()
    finally:
        os.environ["SATQUERY_SHARED_VLM_RUNTIME"] = "1"
        cleanup_shared_vlm()
