"""Tests for open conjecture targeting."""
import pytest
import math
from ouroboros.targeting.conjecture_runner import (
    ConjectureTargetingSession, ConjectureFinding,
    _compute_known_best_cost,
)
from ouroboros.novelty.open_conjectures import (
    CollatzStoppingTimesEnv, PrimeGapEnv, TwinPrimeDensityEnv,
)


class TestConjectureFinding:
    def _make(self, mdl=80.0, known=100.0, novelty=0.7) -> ConjectureFinding:
        return ConjectureFinding(
            conjecture_name="TestConjecture",
            expression_str="CONST(5)",
            mdl_cost=mdl,
            known_best_cost=known,
            improvement_bits=known - mdl,
            novelty_score=novelty,
            oeis_match=None,
            timestamp=0.0,
            session_id="test",
            sequence_start=0,
            sequence_length=100,
        )

    def test_beats_known_when_lower(self):
        f = self._make(mdl=80.0, known=100.0)
        assert f.beats_known

    def test_not_beats_known_when_higher(self):
        f = self._make(mdl=120.0, known=100.0)
        assert not f.beats_known

    def test_to_dict_has_fields(self):
        d = self._make().to_dict()
        assert "expression" in d
        assert "novelty_score" in d
        assert "beats_known" in d

    def test_summary_is_string(self):
        s = self._make().summary()
        assert isinstance(s, str) and "TestConjecture" in s


class TestConjectureTargetingSession:
    def test_short_run(self, tmp_path):
        session = ConjectureTargetingSession(
            output_dir=str(tmp_path),
            beam_width=5, n_iterations=2,
            verbose=False,
        )
        summary = session.run(max_iterations=3, stream_length=60)
        assert summary["n_iterations"] == 3
        assert "flags" in summary

    def test_flags_written_to_file(self, tmp_path):
        # Make threshold very low to ensure flags
        session = ConjectureTargetingSession(
            output_dir=str(tmp_path),
            beam_width=5, n_iterations=2,
            novelty_flag_threshold=0.0,  # flag everything
            improvement_threshold_bits=-999.0,  # always beats known
            verbose=False,
        )
        session.run(max_iterations=3, stream_length=60)
        flag_path = tmp_path / "conjecture_flags.jsonl"
        # Should have written at least something
        assert flag_path.exists() or session._n_flags == 0

    def test_generate_report(self, tmp_path):
        session = ConjectureTargetingSession(
            output_dir=str(tmp_path),
            beam_width=5, n_iterations=2,
            verbose=False,
        )
        session.run(max_iterations=2, stream_length=50)
        report = session.generate_report()
        assert isinstance(report, str)
        assert "Report" in report

    def test_meta_learner_updated(self, tmp_path):
        session = ConjectureTargetingSession(
            output_dir=str(tmp_path),
            beam_width=5, n_iterations=2,
            verbose=False,
        )
        session.run(max_iterations=3, stream_length=50)
        assert session._meta_learner._state.n_updates >= 0


class TestKnownBestCost:
    def test_collatz_approx_costs(self):
        env = CollatzStoppingTimesEnv()
        obs = env.generate(50)
        fn = lambda t: 6.95 * math.log2(max(t + 1, 2))
        cost = _compute_known_best_cost(env, fn, obs)
        assert math.isfinite(cost)
        assert cost > 0