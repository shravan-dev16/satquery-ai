"""Agentic orchestration package for SatQuery AI (Milestone M7)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agent.executor import AgentExecutor, ExecutionError
    from backend.agent.planner import AgentPlanner
    from backend.agent.registry import ModelRegistry, registry
    from backend.agent.router import AgentRouter
    from backend.agent.schema import (
        ExecutionPlan,
        PlanStep,
        RoutingDecision,
        StandardResultContract,
        TaskType,
    )

__all__ = [
    "registry",
    "ModelRegistry",
    "AgentRouter",
    "RoutingDecision",
    "AgentPlanner",
    "ExecutionPlan",
    "PlanStep",
    "AgentExecutor",
    "ExecutionError",
]


def __getattr__(name: str):
    if name in ("AgentExecutor", "ExecutionError"):
        from backend.agent.executor import AgentExecutor, ExecutionError
        return locals()[name]
    if name == "AgentPlanner":
        from backend.agent.planner import AgentPlanner
        return AgentPlanner
    if name == "AgentRouter":
        from backend.agent.router import AgentRouter
        return AgentRouter
    if name in ("ModelRegistry", "registry"):
        from backend.agent.registry import ModelRegistry, registry
        return locals()[name]
    if name in (
        "ExecutionPlan", "PlanStep", "RoutingDecision", "TaskType",
        "ModalityType", "StepStatus", "StandardResultContract",
        "SpecialistInput", "SpecialistOutput", "EvidenceBundle",
        "ImageMetadata", "PairCompatibility", "ValidationResult",
    ):
        import backend.agent.schema as schema
        return getattr(schema, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

