"""Evaluation harnesses and benchmark runner package for SatQuery AI."""

from backend.evaluation.change_benchmark import (
    BenchmarkSampleResult,
    ChangeBenchmarkHarness,
    ChangeBenchmarkReport,
)
from backend.evaluation.failure_taxonomy import (
    FailureCategory,
    FailureDiagnosis,
    FailureTaxonomyClassifier,
)
from backend.evaluation.manifests import (
    EvaluationManifest,
    EvaluationSample,
    SceneCategory,
)
from backend.evaluation.metrics import (
    ChangeMetrics,
    ConfusionMatrix,
    MacroAggregateMetrics,
    PixelGlobalAggregateMetrics,
    aggregate_metrics,
    calculate_confusion_matrix,
    calculate_metrics,
)

__all__ = [
    "BenchmarkSampleResult",
    "ChangeBenchmarkHarness",
    "ChangeBenchmarkReport",
    "ChangeMetrics",
    "ConfusionMatrix",
    "EvaluationManifest",
    "EvaluationSample",
    "FailureCategory",
    "FailureDiagnosis",
    "FailureTaxonomyClassifier",
    "MacroAggregateMetrics",
    "PixelGlobalAggregateMetrics",
    "SceneCategory",
    "aggregate_metrics",
    "calculate_confusion_matrix",
    "calculate_metrics",
]
