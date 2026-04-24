"""Tests for physics law verification system."""
import math
import pytest
from ouroboros.physics.law_signature import (
    PhysicsLaw, _test_hookes_law, _test_exponential_decay,
    _test_free_fall, _safe_corr, _deriv, _deriv2,
)
from ouroboros.physics.derivative_analyzer import (
    DerivativeAnalyzer, PhysicsLawVerifier,
)


class TestDerivativeHelpers:
    def test_deriv_of_linear(self):
        seq = [0.0, 1.0, 2.0, 3.0, 4.0]
        d = _deriv(seq)
        assert all(abs(v - 1.0) < 0.01 for v in d)

    def test_deriv2_of_quadratic(self):
        # t² → DERIV = 2t-1 → DERIV2 = 2 (constant)
        seq = [float(t**2) for t in range(10)]
        d2 = _deriv2(seq)
        assert all(abs(v - 2.0) < 0.01 for v in d2)

    def test_deriv2_of_linear_is_zero(self):
        seq = [float(t) for t in range(10)]
        d2 = _deriv2(seq)
        assert all(abs(v) < 0.01 for v in d2)

    def test_safe_corr_identical(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _safe_corr(x, x) == pytest.approx(1.0)

    def test_safe_corr_opposite(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _safe_corr(x, y) == pytest.approx(-1.0)

    def test_safe_corr_short_returns_zero(self):
        assert _safe_corr([1.0], [1.0]) == 0.0

    def test_safe_corr_empty(self):
        assert _safe_corr([], []) == 0.0


class TestHookesLaw:
    def test_spring_mass_passes(self):
        # x(t) = cos(0.3t) — pure spring motion
        seq = [10.0 * math.cos(0.3 * t) for t in range(100)]
        result = _test_hookes_law(seq, threshold=0.8)
        assert result.passed, f"Hooke's law failed: CORR={result.key_value:.4f}"
        assert result.confidence > 0.5

    def test_monotone_fails(self):
        # Exponential growth is NOT Hooke's law
        seq = [math.exp(0.1 * t) for t in range(50)]
        result = _test_hookes_law(seq, threshold=0.8)
        assert not result.passed

    def test_short_sequence_fails(self):
        result = _test_hookes_law([1.0, 2.0, 3.0], threshold=0.8)
        assert not result.passed

    def test_result_has_law_field(self):
        seq = [math.cos(t) for t in range(50)]
        result = _test_hookes_law(seq)
        assert result.law == PhysicsLaw.HOOKES_LAW


class TestExponentialDecay:
    def test_decay_passes(self):
        seq = [1000.0 * math.exp(-0.05 * t) for t in range(100)]
        result = _test_exponential_decay(seq, threshold=0.8)
        assert result.passed, f"Decay failed: CORR={result.key_value:.4f}"

    def test_oscillation_fails(self):
        seq = [math.sin(t) for t in range(80)]
        result = _test_exponential_decay(seq, threshold=0.8)
        assert not result.passed

    def test_constant_fails(self):
        seq = [5.0] * 50
        result = _test_exponential_decay(seq)
        # Constant has no derivative → correlation undefined → should not pass
        assert not result.passed or result.confidence < 0.5


class TestFreeFall:
    def test_free_fall_passes(self):
        # y = 100 - 4.9*t²  (h=100, g=9.8, dt=1)
        seq = [max(0.0, 100.0 - 4.9 * t**2) for t in range(15)]
        result = _test_free_fall(seq, threshold=0.1)
        assert result.passed, f"Free fall failed: CV={result.key_value:.4f}"

    def test_oscillation_fails_free_fall(self):
        seq = [10.0 * math.sin(t) for t in range(50)]
        result = _test_free_fall(seq, threshold=0.1)
        assert not result.passed


class TestDerivativeAnalyzer:
    def _analyzer(self): return DerivativeAnalyzer()

    def test_spring_is_oscillatory(self):
        seq = [10.0 * math.cos(0.3 * t) for t in range(100)]
        profile = self._analyzer().analyze(seq)
        assert profile.is_oscillatory()

    def test_decay_is_exponential(self):
        seq = [1000.0 * math.exp(-0.1 * t) for t in range(60)]
        profile = self._analyzer().analyze(seq)
        assert profile.is_exponential()

    def test_free_fall_constant_accel(self):
        seq = [max(0.0, 100.0 - 4.9 * t**2) for t in range(15)]
        profile = self._analyzer().analyze(seq)
        assert profile.is_constant_acceleration()

    def test_identify_spring_law(self):
        seq = [10.0 * math.cos(0.3 * t) for t in range(100)]
        law, results = self._analyzer().identify_law(seq)
        assert law in (PhysicsLaw.HOOKES_LAW, PhysicsLaw.SIMPLE_HARMONIC)

    def test_identify_decay_law(self):
        seq = [1000.0 * math.exp(-0.05 * t) for t in range(80)]
        law, results = self._analyzer().identify_law(seq)
        assert law in (PhysicsLaw.EXPONENTIAL_DECAY, PhysicsLaw.NEWTON_COOLING)

    def test_empty_profile_no_crash(self):
        profile = self._analyzer().analyze([1.0, 2.0])
        assert isinstance(profile.corr_d2_orig, float)


class TestPhysicsLawVerifier:
    def test_spring_env_passes_hookes_law(self):
        from ouroboros.environments.physics import SpringMassEnv
        verifier = PhysicsLawVerifier()
        env = SpringMassEnv(amplitude=10, omega=0.3, as_integer=False)
        seq = env.generate(100)
        law, results = verifier.verify_raw_sequence([float(v) for v in seq])
        assert law in (PhysicsLaw.HOOKES_LAW, PhysicsLaw.SIMPLE_HARMONIC)

    def test_decay_env_passes_decay_law(self):
        from ouroboros.environments.physics import RadioactiveDecayEnv
        verifier = PhysicsLawVerifier()
        env = RadioactiveDecayEnv(n0=1000, decay_rate=0.05)
        seq = env.generate(100)
        law, results = verifier.verify_raw_sequence([float(v) for v in seq])
        assert law in (PhysicsLaw.EXPONENTIAL_DECAY, PhysicsLaw.NEWTON_COOLING)

    def test_free_fall_passes(self):
        from ouroboros.environments.physics import FreeFallEnv
        verifier = PhysicsLawVerifier()
        env = FreeFallEnv(h0=100, g=9.8, scale=0.05)
        seq = env.generate(20)
        law, results = verifier.verify_raw_sequence([float(v) for v in seq])
        assert law in (PhysicsLaw.FREE_FALL, PhysicsLaw.UNKNOWN)

    def test_lean4_generator_produces_code(self):
        from ouroboros.physics.discovery_runner import generate_hookes_law_lean4
        code = generate_hookes_law_lean4(spring_constant_estimate=0.09)
        assert "theorem" in code
        assert "Hooke" in code or "harmonic" in code
        assert "sorry" not in code.lower()