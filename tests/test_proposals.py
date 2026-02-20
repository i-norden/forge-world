"""Tests for structured parameter proposals."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from forge_world.core.proposals import (
    ParameterProposal,
    ProposalFile,
    apply_proposals,
    rollback_proposals,
)
from tests.test_runner import FakePipeline


class TestParameterProposal:
    def test_round_trip(self):
        p = ParameterProposal(
            parameter_path="weight_ela",
            new_value=0.75,
            reasoning="High sensitivity impact",
        )
        d = p.to_dict()
        restored = ParameterProposal.from_dict(d)
        assert restored.parameter_path == "weight_ela"
        assert restored.new_value == 0.75
        assert restored.reasoning == "High sensitivity impact"

    def test_no_reasoning(self):
        p = ParameterProposal(parameter_path="threshold", new_value=0.6)
        d = p.to_dict()
        assert "reasoning" not in d
        restored = ParameterProposal.from_dict(d)
        assert restored.reasoning == ""


class TestProposalFile:
    def test_round_trip(self):
        pf = ProposalFile(
            proposals=[
                ParameterProposal("weight_ela", 0.75, "test"),
                ParameterProposal("threshold", 0.6),
            ],
            agent_notes="Focusing on top-2 parameters",
        )
        d = pf.to_dict()
        restored = ProposalFile.from_dict(d)
        assert len(restored.proposals) == 2
        assert restored.agent_notes == "Focusing on top-2 parameters"

    def test_save_load(self):
        pf = ProposalFile(
            proposals=[ParameterProposal("threshold", 0.6)],
            agent_notes="Test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "proposal.json"
            pf.save(path)
            loaded = ProposalFile.load(path)
            assert loaded is not None
            assert len(loaded.proposals) == 1
            assert loaded.proposals[0].parameter_path == "threshold"

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            assert ProposalFile.load(path) is None


class TestApplyProposals:
    def test_apply_flat(self):
        pipeline = FakePipeline(config={"threshold": 0.5})
        old = apply_proposals(
            pipeline,
            [ParameterProposal("threshold", 0.6)],
        )
        assert pipeline.get_config()["threshold"] == 0.6
        assert old["threshold"] == 0.5

    def test_apply_nested(self):
        pipeline = FakePipeline(config={"threshold": 0.5, "ela": {"quality": 80}})
        old = apply_proposals(
            pipeline,
            [ParameterProposal("ela.quality", 85)],
        )
        assert pipeline.get_config()["ela"]["quality"] == 85
        assert old["ela"]["quality"] == 80

    def test_returns_old_config(self):
        pipeline = FakePipeline(config={"threshold": 0.5})
        old = apply_proposals(
            pipeline,
            [ParameterProposal("threshold", 0.6)],
        )
        assert old["threshold"] == 0.5
        # Old config should not be mutated by subsequent changes
        pipeline.set_config({"threshold": 0.7})
        assert old["threshold"] == 0.5

    def test_unknown_path_raises(self):
        pipeline = FakePipeline(config={"threshold": 0.5})
        with pytest.raises(KeyError, match="Unknown parameter path"):
            apply_proposals(
                pipeline,
                [ParameterProposal("nonexistent_param", 0.6)],
            )

    def test_rollback(self):
        pipeline = FakePipeline(config={"threshold": 0.5})
        old = apply_proposals(
            pipeline,
            [ParameterProposal("threshold", 0.6)],
        )
        assert pipeline.get_config()["threshold"] == 0.6
        rollback_proposals(pipeline, old)
        assert pipeline.get_config()["threshold"] == 0.5

    def test_proposal_file_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "proposal.json"
            pf = ProposalFile(
                proposals=[ParameterProposal("threshold", 0.6)],
            )
            pf.save(path)
            assert path.exists()
            path.unlink(missing_ok=True)
            assert not path.exists()


class TestValidateProposals:
    """Tests for schema-based proposal validation."""

    def _schema(self):
        return {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }

    def _config(self):
        return {"threshold": 0.5, "count": 10, "weight": 0.7}

    def test_valid_proposal_passes(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("threshold", 0.6)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 1
        assert len(result.errors) == 0

    def test_unknown_path_rejected(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("nonexistent_param", 0.6)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 0
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "unknown_path"

    def test_out_of_range_rejected(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("threshold", 1.5)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 0
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "out_of_range"

    def test_wrong_type_rejected(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("count", 3.7)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 0
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "wrong_type"

    def test_mixed_valid_and_invalid(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [
            ParameterProposal("threshold", 0.6),   # valid
            ParameterProposal("threshold", 1.5),    # out of range
            ParameterProposal("weight", 0.8),       # valid
        ]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 2
        assert len(result.errors) == 1

    def test_error_summary_format(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [
            ParameterProposal("nonexistent", 0.6),
            ParameterProposal("threshold", 2.0),
        ]
        result = validate_proposals(proposals, self._schema(), self._config())
        summary = result.error_summary()
        assert "## Proposal Validation Errors" in summary
        assert "nonexistent" in summary
        assert "threshold" in summary

    def test_no_errors_empty_summary(self):
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("threshold", 0.6)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert result.error_summary() == ""

    def test_integer_accepts_whole_float(self):
        """A float like 5.0 should be accepted for integer params."""
        from forge_world.core.proposals import validate_proposals
        proposals = [ParameterProposal("count", 5.0)]
        result = validate_proposals(proposals, self._schema(), self._config())
        assert len(result.valid_proposals) == 1
        assert len(result.errors) == 0
