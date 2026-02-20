"""Structured parameter proposals for the evolution loop.

Provides a fast parameter-only tuning path: the agent writes a JSON
proposal file specifying parameter changes, and the evolution loop applies
them programmatically without requiring source file edits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_world.core.sensitivity import (
    get_nested_value,
    set_nested_value,
    walk_numeric_parameters,
)


@dataclass
class ParameterProposal:
    """A single parameter change proposed by the agent."""

    parameter_path: str  # "weight_ela" or "ela.quality"
    new_value: Any
    reasoning: str = ""  # agent's justification

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "parameter_path": self.parameter_path,
            "new_value": self.new_value,
        }
        if self.reasoning:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterProposal:
        return cls(
            parameter_path=data["parameter_path"],
            new_value=data["new_value"],
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class ProposalFile:
    """Contents of .forge-world/parameter-proposal.json"""

    proposals: list[ParameterProposal]
    agent_notes: str = ""  # high-level reasoning

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "proposals": [p.to_dict() for p in self.proposals],
        }
        if self.agent_notes:
            d["agent_notes"] = self.agent_notes
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposalFile:
        return cls(
            proposals=[ParameterProposal.from_dict(p) for p in data.get("proposals", [])],
            agent_notes=data.get("agent_notes", ""),
        )

    @classmethod
    def load(cls, path: Path) -> ProposalFile | None:
        """Load from disk. Returns None if file missing."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def apply_proposals(pipeline: Any, proposals: list[ParameterProposal]) -> dict[str, Any]:
    """Apply parameter proposals to pipeline config.

    1. Get current config via pipeline.get_config()
    2. Serialize to dict
    3. For each proposal, set value at parameter_path
    4. Apply via pipeline.set_config()
    5. Return the previous config dict (for rollback)

    Raises KeyError if a parameter_path doesn't exist in the config.
    """
    old_config = pipeline.get_config()
    if not isinstance(old_config, dict):
        old_config = (
            old_config.model_dump() if hasattr(old_config, "model_dump") else dict(old_config)
        )
    old_config_copy = _deep_copy_dict(old_config)

    new_config = _deep_copy_dict(old_config)
    for proposal in proposals:
        try:
            # Verify path exists (raises KeyError/AttributeError if not)
            get_nested_value(new_config, proposal.parameter_path)
        except (KeyError, AttributeError) as exc:
            raise KeyError(f"Unknown parameter path: {proposal.parameter_path}") from exc
        set_nested_value(new_config, proposal.parameter_path, proposal.new_value)

    pipeline.set_config(new_config)
    return old_config_copy


def rollback_proposals(pipeline: Any, old_config: dict[str, Any]) -> None:
    """Restore pipeline config to previous state via set_config()."""
    pipeline.set_config(old_config)


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Deep copy a nested dict (avoids importing copy for simple case)."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


# --- Schema Validation ---


@dataclass
class ProposalValidationError:
    """A single validation error for a parameter proposal."""

    parameter_path: str
    error_type: str  # "unknown_path" | "out_of_range" | "wrong_type"
    message: str
    proposed_value: Any


@dataclass
class ProposalValidationResult:
    """Result of validating a list of proposals against a schema."""

    valid_proposals: list[ParameterProposal] = field(default_factory=list)
    errors: list[ProposalValidationError] = field(default_factory=list)

    def error_summary(self) -> str:
        """Markdown summary of validation errors for agent context."""
        if not self.errors:
            return ""
        lines = ["## Proposal Validation Errors", ""]
        for err in self.errors:
            lines.append(
                f"- **{err.parameter_path}**: {err.error_type} — {err.message} "
                f"(proposed: {err.proposed_value})"
            )
        lines.append("")
        return "\n".join(lines)


def validate_proposals(
    proposals: list[ParameterProposal],
    schema: dict[str, Any],
    current_config: dict[str, Any],
) -> ProposalValidationResult:
    """Validate proposals against a JSON schema and current config.

    Checks each proposal:
    - Path exists in schema or current config
    - Value is within [minimum, maximum] bounds from schema
    - Type matches (integer vs number)
    """
    # Build lookup of schema parameters by path
    schema_params = walk_numeric_parameters(schema)
    schema_by_path: dict[str, dict[str, Any]] = {p["path"]: p for p in schema_params}

    result = ProposalValidationResult()

    for proposal in proposals:
        path = proposal.parameter_path
        value = proposal.new_value

        # Check if path exists
        param_info = schema_by_path.get(path)
        if param_info is None:
            # Also check if it exists in current config (non-numeric params)
            try:
                get_nested_value(current_config, path)
            except (KeyError, AttributeError):
                result.errors.append(
                    ProposalValidationError(
                        parameter_path=path,
                        error_type="unknown_path",
                        message=f"Parameter '{path}' not found in schema or config",
                        proposed_value=value,
                    )
                )
                continue
            # Path exists in config but not in schema — allow it (non-numeric)
            result.valid_proposals.append(proposal)
            continue

        # Check type
        param_type = param_info.get("type", "number")
        if param_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                # Allow float if it's a whole number
                if isinstance(value, float) and value == int(value):
                    pass  # acceptable
                else:
                    result.errors.append(
                        ProposalValidationError(
                            parameter_path=path,
                            error_type="wrong_type",
                            message=f"Expected integer, got {type(value).__name__}",
                            proposed_value=value,
                        )
                    )
                    continue

        # Check bounds
        param_min = param_info.get("minimum")
        param_max = param_info.get("maximum")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            result.errors.append(
                ProposalValidationError(
                    parameter_path=path,
                    error_type="wrong_type",
                    message=f"Expected numeric value, got {type(value).__name__}",
                    proposed_value=value,
                )
            )
            continue

        if param_min is not None and numeric_value < float(param_min):
            result.errors.append(
                ProposalValidationError(
                    parameter_path=path,
                    error_type="out_of_range",
                    message=f"Value {value} below minimum {param_min}",
                    proposed_value=value,
                )
            )
            continue

        if param_max is not None and numeric_value > float(param_max):
            result.errors.append(
                ProposalValidationError(
                    parameter_path=path,
                    error_type="out_of_range",
                    message=f"Value {value} above maximum {param_max}",
                    proposed_value=value,
                )
            )
            continue

        result.valid_proposals.append(proposal)

    return result
