"""
LongRangeBeamSearch with environment classes and BM warmstart.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ouroboros.environments.base import ObservationEnvironment
from ouroboros.synthesis.expr_node import ExprNode, NodeType
from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
from ouroboros.compression.mdl_engine import MDLEngine


# ?? Recurrence Environments ???????????????????????????????????????????????????

class TribonacciModEnv:
    def __init__(self, modulus=7, seeds=(0, 1, 1)):
        self.modulus = modulus
        self.seeds = tuple(seeds)
        self.max_lag = 3

    def generate(self, length: int) -> List[int]:
        s = list(self.seeds[:3])
        while len(s) < 3: s.append(0)
        seq = list(s)
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2] + seq[-3]) % self.modulus)
        return seq[:length]

    def _generate_stream(self):
        s = list(self.seeds[:3])
        while len(s) < 3: s.append(0)
        while True:
            yield s[0]
            s = [s[1], s[2], (s[0] + s[1] + s[2]) % self.modulus]

    @property
    def name(self): return f"Tribonacci({self.modulus})"

    def ground_truth_rule_str(self):
        return f"(prev(1)+prev(2)+prev(3)) mod {self.modulus}"


class LucasSequenceEnv:
    def __init__(self, modulus=7, seeds=(2, 1)):
        self.modulus = modulus
        self.seeds = tuple(seeds)
        self.max_lag = 2

    def generate(self, length: int) -> List[int]:
        a, b = self.seeds
        seq = [a % self.modulus, b % self.modulus]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % self.modulus)
        return seq[:length]

    def _generate_stream(self):
        a, b = self.seeds[0] % self.modulus, self.seeds[1] % self.modulus
        while True:
            yield a
            a, b = b, (a + b) % self.modulus

    @property
    def name(self): return f"Lucas({self.modulus})"


class LinearRecurrenceEnv:
    def __init__(self, coefficients, modulus=7, seeds=None):
        self.coefficients = list(coefficients)
        self.modulus = modulus
        order = len(coefficients)
        self.seeds = list(seeds) if seeds else [0] * order
        while len(self.seeds) < order: self.seeds.append(0)
        self.max_lag = order

    def generate(self, length: int) -> List[int]:
        order = len(self.coefficients)
        buf = list(self.seeds[:order])
        seq = list(buf)
        while len(seq) < length:
            nxt = sum(c * buf[i] for i, c in enumerate(self.coefficients)) % self.modulus
            buf = buf[1:] + [nxt]
            seq.append(nxt)
        return seq[:length]

    def _generate_stream(self):
        order = len(self.coefficients)
        buf = list(self.seeds[:order])
        for v in buf: yield v
        while True:
            nxt = sum(c * buf[i] for i, c in enumerate(self.coefficients)) % self.modulus
            buf = buf[1:] + [nxt]
            yield nxt

    @property
    def name(self): return f"LinearRecurrence({self.modulus})"


class AutoregressiveEnv:
    def __init__(self, nonzero_lags, modulus=7):
        self.nonzero_lags = list(nonzero_lags)
        self.modulus = modulus
        self.max_lag = max(lag for lag, _ in self.nonzero_lags)

    def generate(self, length: int) -> List[int]:
        buf = [0] * self.max_lag
        seq = list(buf)
        while len(seq) < length:
            nxt = sum(c * buf[-lag] for lag, c in self.nonzero_lags) % self.modulus
            buf = buf[1:] + [nxt]
            seq.append(nxt)
        return seq[:length]

    def _generate_stream(self):
        buf = [0] * self.max_lag
        for v in buf: yield v
        while True:
            nxt = sum(c * buf[-lag] for lag, c in self.nonzero_lags) % self.modulus
            buf = buf[1:] + [nxt]
            yield nxt

    @property
    def name(self): return f"Autoregressive({self.modulus})"


# ?? Berlekamp-Massey ??????????????????????????????????????????????????????????

class BerlekampMassey:
    """Berlekamp-Massey algorithm for finding minimal linear recurrence."""

    def run(self, seq: List[int], modulus: int) -> Optional[List[int]]:
        """Return coefficients [c1..ck] s.t. s[n] = sum(ci*s[n-i]) mod p, or None."""
        n = len(seq)
        C, B = [1], [1]
        L, m, b = 0, 1, 1
        for i in range(n):
            d = seq[i]
            for j in range(1, L + 1):
                if j < len(C):
                    d = (d + C[j] * seq[i - j]) % modulus
            if d == 0:
                m += 1
            elif 2 * L <= i:
                T = list(C)
                coef = d * pow(b, modulus - 2, modulus) % modulus
                while len(C) < len(B) + m:
                    C.append(0)
                for j in range(len(B)):
                    C[j + m] = (C[j + m] - coef * B[j]) % modulus
                L, B, b, m = i + 1 - L, T, d, 1
            else:
                coef = d * pow(b, modulus - 2, modulus) % modulus
                while len(C) < len(B) + m:
                    C.append(0)
                for j in range(len(B)):
                    C[j + m] = (C[j + m] - coef * B[j]) % modulus
                m += 1
        if L == 0:
            return None
        # C[0]=1, coefficients are -C[1..]
        coeffs = [(-C[i]) % modulus for i in range(1, L + 1)]
        return coeffs


@dataclass
class RecurrenceAxiom:
    coefficients: List[int]
    modulus: int
    order: int
    fit_error: float
    expression_str: str


class RecurrenceDetector:
    def __init__(self, max_order=20, accuracy_threshold=0.95):
        self.max_order = max_order
        self.accuracy_threshold = accuracy_threshold
        self._bm = BerlekampMassey()

    def detect(self, seq: List[int], modulus: int, env_name: str = "") -> Optional[RecurrenceAxiom]:
        if modulus < 2:
            return None
        # Check modulus is prime (BM needs GF(p))
        if not self._is_prime(modulus):
            return None
        coeffs = self._bm.run(seq, modulus)
        if coeffs is None or len(coeffs) > self.max_order:
            return None
        # Validate
        order = len(coeffs)
        errors = 0
        for i in range(order, len(seq)):
            pred = sum(coeffs[j] * seq[i - j - 1] for j in range(order)) % modulus
            if pred != seq[i]:
                errors += 1
        fit_error = errors / max(1, len(seq) - order)
        if (1 - fit_error) < self.accuracy_threshold:
            return None
        expr_str = "(" + "+".join(f"{c}*prev({j+1})" for j, c in enumerate(coeffs) if c) + f") mod {modulus}"
        return RecurrenceAxiom(coefficients=coeffs, modulus=modulus, order=order,
                               fit_error=fit_error, expression_str=expr_str)

    def _is_prime(self, n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True


# ?? LongRangeBeamSearch ???????????????????????????????????????????????????????

@dataclass
class LongRangeBeamConfig:
    beam_width: int = 25
    max_lag: int = 20
    const_range: int = 30
    max_depth: int = 5
    mcmc_iterations: int = 150
    use_bm_warmstart: bool = True
    bm_accuracy_threshold: float = 0.95
    random_seed: int = 42


@dataclass
class LongRangeResult:
    best_expr: Optional[ExprNode]
    best_mdl_cost: float
    discovery_method: str
    recurrence_axiom: Optional[RecurrenceAxiom] = None


def recurrence_to_expr(axiom: RecurrenceAxiom) -> Optional[ExprNode]:
    nonzero = [(i+1, c) for i, c in enumerate(axiom.coefficients) if c != 0]
    if not nonzero:
        return ExprNode(NodeType.CONST, value=0)

    def make_term(lag, coeff):
        prev_node = ExprNode(NodeType.PREV, lag=lag)
        if coeff == 1:
            return prev_node
        return ExprNode(NodeType.MUL,
            left=ExprNode(NodeType.CONST, value=coeff), right=prev_node)

    terms = [make_term(lag, c) for lag, c in nonzero]
    acc = terms[0]
    for term in terms[1:]:
        acc = ExprNode(NodeType.ADD, left=acc, right=term)
    return ExprNode(NodeType.MOD, left=acc,
                    right=ExprNode(NodeType.CONST, value=axiom.modulus))


class LongRangeBeamSearch:
    def __init__(self, config: LongRangeBeamConfig = None):
        self.cfg = config or LongRangeBeamConfig()
        self._detector = RecurrenceDetector(
            max_order=self.cfg.max_lag,
            accuracy_threshold=self.cfg.bm_accuracy_threshold,
        )
        self._mdl = MDLEngine()

    def search(self, observations: List[int], modulus: int,
               environment_name: str = "unknown", verbose: bool = False) -> LongRangeResult:
        if self.cfg.use_bm_warmstart and len(observations) >= 50:
            axiom = self._detector.detect(observations, modulus, environment_name)
            if axiom is not None:
                expr = recurrence_to_expr(axiom)
                if expr is not None:
                    cost = self._score(expr, observations)
                    return LongRangeResult(best_expr=expr, best_mdl_cost=cost,
                                           discovery_method="BerlekampMassey",
                                           recurrence_axiom=axiom)
        beam_cfg = BeamConfig(
            beam_width=self.cfg.beam_width,
            const_range=self.cfg.const_range,
            max_depth=self.cfg.max_depth,
            max_lag=self.cfg.max_lag,
            mcmc_iterations=self.cfg.mcmc_iterations,
            random_seed=self.cfg.random_seed,
        )
        synthesizer = BeamSearchSynthesizer(beam_cfg)
        best_expr = synthesizer.search(observations)
        if best_expr is None:
            return LongRangeResult(best_expr=None, best_mdl_cost=float("inf"),
                                   discovery_method="BeamSearch")
        cost = self._score(best_expr, observations)
        return LongRangeResult(best_expr=best_expr, best_mdl_cost=cost,
                               discovery_method="BeamSearch")

    def _score(self, expr: ExprNode, observations: List[int]) -> float:
        predictions = [expr.evaluate(t, observations[:t]) for t in range(len(observations))]
        result = self._mdl.compute(predictions, observations,
                                   expr.node_count(), expr.constant_count())
        return result.total_mdl_cost