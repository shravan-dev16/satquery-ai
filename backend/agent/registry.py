"""Central Model and Tool Registry.

Implements AGENTS.md Rule 5 (Specialist Model Architecture) providing
registration, discovery, capability inspection, and execution dispatching
for remote-sensing specialists.
"""

import logging
from typing import Dict, List, Optional

from backend.agent.schema import ModalityType, ModelCapability, TaskType
from backend.models.base import BaseSpecialist

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry tracking available specialist models and preprocessors."""

    def __init__(self) -> None:
        self._specialists: Dict[str, BaseSpecialist] = {}

    def register(self, specialist: BaseSpecialist) -> None:
        """Registers a specialist model with the central registry."""
        cap = specialist.capability
        if cap.identifier in self._specialists:
            logger.warning("Overwriting previously registered specialist: %s", cap.identifier)
        self._specialists[cap.identifier] = specialist
        logger.info("Registered specialist [%s] for task [%s]", cap.identifier, cap.task)

    def unregister(self, identifier: str) -> Optional[BaseSpecialist]:
        """Removes a specialist from the registry."""
        return self._specialists.pop(identifier, None)

    def get(self, identifier: str) -> Optional[BaseSpecialist]:
        """Retrieves a registered specialist by its unique identifier."""
        return self._specialists.get(identifier)

    def list_capabilities(self) -> List[ModelCapability]:
        """Returns capabilities of all currently registered specialists."""
        return [spec.capability for spec in self._specialists.values()]

    def find_specialists(
        self,
        task: TaskType,
        modality: Optional[ModalityType] = None,
        input_count: Optional[int] = None,
    ) -> List[BaseSpecialist]:
        """Finds matching specialists by task type, input modality, and image count."""
        matches: List[BaseSpecialist] = []
        for spec in self._specialists.values():
            cap = spec.capability
            if cap.task != task:
                continue
            if modality and modality not in cap.supported_modalities:
                continue
            if input_count is not None and input_count not in cap.supported_input_count:
                continue
            matches.append(spec)
        return matches

    def health_check_all(self) -> Dict[str, bool]:
        """Runs health checks on all registered specialists."""
        return {
            ident: spec.health_check()
            for ident, spec in self._specialists.items()
        }

    def clear(self) -> None:
        """Clears all registered specialists (useful for isolated unit testing)."""
        self._specialists.clear()


# Global singleton instance
registry = ModelRegistry()
