"""Core modules for the forge-world evolution harness."""

from forge_world.core.agent_interface import (
    EvolutionContext,
    build_evolution_context,
)
from forge_world.core.evolve import EvolutionConfig, EvolutionLoop, EvolutionResult
from forge_world.core.memory import (
    EvolutionMemory,
    MemoryEntry,
    ParameterChange,
    compute_parameter_diff,
)
from forge_world.core.proposals import (
    ParameterProposal,
    ProposalFile,
    ProposalValidationError,
    ProposalValidationResult,
    apply_proposals,
    rollback_proposals,
    validate_proposals,
)
from forge_world.core.runner import PerformanceMetrics
from forge_world.core.sensitivity import (
    ParameterSensitivity,
    SensitivityReport,
    compute_sensitivity,
)

__all__ = [
    # agent_interface
    "EvolutionContext",
    "build_evolution_context",
    # evolve
    "EvolutionConfig",
    "EvolutionLoop",
    "EvolutionResult",
    # memory
    "EvolutionMemory",
    "MemoryEntry",
    "ParameterChange",
    "compute_parameter_diff",
    # proposals
    "ParameterProposal",
    "ProposalFile",
    "ProposalValidationError",
    "ProposalValidationResult",
    "apply_proposals",
    "rollback_proposals",
    "validate_proposals",
    # runner
    "PerformanceMetrics",
    # sensitivity
    "ParameterSensitivity",
    "SensitivityReport",
    "compute_sensitivity",
]
