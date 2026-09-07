"""Unit tests for text-guided region grounding specialist and coordinate parser."""

import pytest
from pathlib import Path
import torch

from backend.agent.registry import ModelRegistry
from backend.agent.schema import ModalityType, SpecialistInput, TaskType
from backend.evidence.spatial import PixelBoundingBox
from backend.models.grounding import (
    GroundingCoordinateParser,
    GroundingDinoSpecialist,
    QwenGroundingSpecialist,
    RemoteSensingGroundingSpecialist,
)


def test_grounding_specialist_capability():
    """Verify grounding specialist implements BaseSpecialist and registers with ModelRegistry."""
    specialist = RemoteSensingGroundingSpecialist()
    cap = specialist.capability
    assert cap.identifier == "RS_GROUND"
    assert cap.task == TaskType.GROUNDING
    assert ModalityType.OPTICAL in cap.supported_modalities

    reg = ModelRegistry()
    reg.register(specialist)
    assert reg.get("RS_GROUND") is specialist
    matches = reg.find_specialists(task=TaskType.GROUNDING, modality=ModalityType.OPTICAL, input_count=1)
    assert len(matches) == 1
    assert matches[0] is specialist


def test_coordinate_parser_qwen2_vl_json_format():
    """Verify parser extracts Qwen2-VL normalized [0, 1000] box_2d format."""
    model_output = 'Here is the location: {"box_2d": [200, 100, 600, 500], "label": "water body"}'
    boxes = GroundingCoordinateParser.parse_boxes(
        text=model_output,
        image_width=1000,
        image_height=1000,
        target_label="water body",
    )
    assert len(boxes) == 1
    box = boxes[0]
    # In Qwen2-VL: [ymin=200, xmin=100, ymax=600, xmax=500]
    # Expected pixel bbox: [xmin=100, ymin=200, xmax=500, ymax=600]
    assert box.xmin == 100.0
    assert box.ymin == 200.0
    assert box.xmax == 500.0
    assert box.ymax == 600.0
    assert box.label == "water body"


def test_coordinate_parser_normalized_0_1_format():
    """Verify parser extracts [0.0, 1.0] normalized bounding box format."""
    model_output = "Detected runway at [0.1, 0.2, 0.8, 0.6]"
    boxes = GroundingCoordinateParser.parse_boxes(
        text=model_output,
        image_width=200,
        image_height=100,
        target_label="runway",
    )
    assert len(boxes) == 1
    box = boxes[0]
    # xmin=0.1*200=20, ymin=0.2*100=20, xmax=0.8*200=160, ymax=0.6*100=60
    assert box.xmin == 20.0
    assert box.ymin == 20.0
    assert box.xmax == 160.0
    assert box.ymax == 60.0


def test_coordinate_parser_empty_or_malformed():
    """Verify empty or non-coordinate text returns empty list without error."""
    text_no_boxes = "I cannot find any aircraft in this image."
    boxes = GroundingCoordinateParser.parse_boxes(
        text=text_no_boxes,
        image_width=256,
        image_height=256,
    )
    assert len(boxes) == 0


def test_coordinate_parser_box_tokens():
    """Verify parser extracts Qwen2-VL <|box_start|>(ymin, xmin),(ymax, xmax)<|box_end|> tokens."""
    model_output = "The runway is located at <|box_start|>(200, 100),(800, 500)<|box_end|>."
    boxes = GroundingCoordinateParser.parse_boxes(
        text=model_output,
        image_width=1000,
        image_height=1000,
        target_label="runway",
    )
    assert len(boxes) == 1
    box = boxes[0]
    # Qwen2-VL: ymin=200, xmin=100, ymax=800, xmax=500
    # Expected: xmin=100, ymin=200, xmax=500, ymax=800
    assert box.xmin == 100.0
    assert box.ymin == 200.0
    assert box.xmax == 500.0
    assert box.ymax == 800.0
    assert box.label == "runway"


def test_grounding_pipeline_empty_result(optical_sample_path: Path):
    """Verify warning returned when model produces no bounding boxes."""
    specialist = QwenGroundingSpecialist()

    class MockEmptyModel:
        def generate(self, **kwargs):
            class Out:
                sequences = [[1, 2, 3]]
                scores = None
            return Out()

    class MockEmptyProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"
        def __call__(self, *args, **kwargs):
            class TensorDict(dict):
                def __init__(self):
                    super().__init__(input_ids=[[1, 2]])
                    self.input_ids = [[1, 2]]
                def to(self, *args): return self
            return TensorDict()
        def batch_decode(self, *args, **kwargs):
            return ["I cannot find any such feature in this image."]

    specialist.model = MockEmptyModel()
    specialist.processor = MockEmptyProcessor()
    specialist._is_loaded = True

    inputs = SpecialistInput(
        task=TaskType.GROUNDING,
        query="Locate the aircraft.",
        primary_image_path=str(optical_sample_path),
    )

    output = specialist.predict(inputs)
    assert output.success is True
    assert len(output.evidence.boxes) == 0
    assert len(output.warnings) >= 1
    assert "No valid grounded regions were detected" in output.warnings[0]
    assert output.confidence == 0.0


def test_grounding_dino_pipeline_with_mock(optical_sample_path: Path):
    """Verify Grounding DINO specialist end-to-end pipeline with mock outputs."""
    specialist = GroundingDinoSpecialist(device="cpu")

    class MockGDModel:
        def __call__(self, **kwargs):
            return {}

    class MockGDProcessor:
        def __call__(self, *args, **kwargs):
            class OutDict(dict):
                def __init__(self):
                    super().__init__(input_ids=torch.tensor([[1, 2]]))
                    self.input_ids = torch.tensor([[1, 2]])
                def to(self, *args): return self
            return OutDict()

        def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
            return [{
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
                "scores": torch.tensor([0.88]),
                "text_labels": ["water body"],
            }]

    specialist.model = MockGDModel()
    specialist.processor = MockGDProcessor()
    specialist._is_loaded = True

    inputs = SpecialistInput(
        task=TaskType.GROUNDING,
        query="Where is the water body?",
        primary_image_path=str(optical_sample_path),
    )

    output = specialist.predict(inputs)
    assert output.success is True
    assert len(output.evidence.boxes) == 1
    assert len(output.evidence.regions) == 1
    assert output.evidence.boxes[0].label == "water body"
    assert output.evidence.boxes[0].model_score == 0.88
    assert output.evidence.boxes[0].is_degenerate is False
    assert output.evidence.boxes[0].geojson is not None
    assert output.confidence == 0.88


def test_grounding_pipeline_with_mock_model(optical_sample_path: Path):
    """Verify grounding pipeline end-to-end with mock model output and regions contract."""
    specialist = QwenGroundingSpecialist()

    # Mock processor and model behavior
    class MockModel:
        def generate(self, **kwargs):
            class Out:
                sequences = [[1, 2, 3]]
                scores = None
            return Out()

    class MockProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"
        def __call__(self, *args, **kwargs):
            class TensorDict(dict):
                def __init__(self):
                    super().__init__(input_ids=[[1, 2]])
                    self.input_ids = [[1, 2]]
                def to(self, *args): return self
            return TensorDict()
        def batch_decode(self, *args, **kwargs):
            return ['{"box_2d": [100, 100, 500, 500], "label": "water body"}']

    specialist.model = MockModel()
    specialist.processor = MockProcessor()
    specialist._is_loaded = True

    inputs = SpecialistInput(
        task=TaskType.GROUNDING,
        query="Where is the water body?",
        primary_image_path=str(optical_sample_path),
    )

    output = specialist.predict(inputs)
    assert output.success is True
    assert len(output.evidence.boxes) == 1
    assert len(output.evidence.regions) == 1

    box = output.evidence.boxes[0]
    assert box.label == "water body"
    assert box.geojson is not None
    assert box.geojson["type"] == "Polygon"
    assert len(box.geojson["coordinates"][0]) == 5

    # Check region contract
    region = output.evidence.regions[0]
    assert region.bbox_pixel is not None
    assert region.polygon_pixel is None  # Axis-aligned bbox, not a segmentation mask
    assert region.confidence == box.confidence
