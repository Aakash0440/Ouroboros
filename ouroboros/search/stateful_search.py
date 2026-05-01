"""
StatefulSearch — Beam search that carries STATE_VAR across timesteps.

The fundamental change from all previous search code:
  Before: score(expr, obs) = evaluate expr at each t independently
  After:  score(expr, obs, state) = evaluate expr sequentially, state persists

This is what enables algorithm discovery rather than formula discovery.
A formula computes obs[t] from t and history.
An algorithm computes obs[t] from t, history, AND accumulated state.

For GCDEnv: the Euclidean algorithm maintains state[0]=a, state[1]=b
and at each step computes: new_b = a % b, new_a = b, state = {0:new_a, 1:new_b}
The output is state[0] when state[1]=0.

With STATE nodes this is representable. Without them it is not.

The cost of stateful evaluation:
  Stateless: O(beam_width × n_timesteps) evaluations, all independent
  Stateful: O(beam_width × n_timesteps) evaluations, SEQUENTIAL per expression
  The stateful evaluations cannot be parallelized across timesteps
  (each timestep depends on the previous one's state)
  → stateful search is slower by a constant factor but not asymptotically worse

Implementation:
  StatefulBeamSearch wraps GrammarConstrainedBeam
  - Scoring is now sequential (not vectorized)
  - STATE_VAR nodes read/write the persistent state dict
  - Each beam candidate gets its own fresh state dict per scoring
"""

from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType, NODE_SPECS, NodeCategory
from ouroboros.grammar.math_grammar import DEFAULT_GRAMMAR
from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
from ouroboros.compression.mdl_engine import MDLEngine


@dataclass
class StatefulSearchConfig:
    """Configuration for stateful beam search."""
    beam_width: int = 20
    max_depth: int = 5
    const_range: int = 30
    max_lag: int = 5
    n_iterations: int = 12
    n_state_vars: int = 4         # number of STATE_VAR slots available (0..n-1)
    initial_state: Dict[int, float] = field(default_factory=dict)
    random_seed: int = 42
    verbose: bool = False


@dataclass
class StatefulCandidate:
    """A beam candidate with its sequential MDL score."""
    expr: ExtExprNode
    mdl_cost: float
    final_state: Dict[int, float]   # state after full sequence evaluation

    def __lt__(self, other):
        return self.mdl_cost < other.mdl_cost


class StatefulScorer:
    """
    Scores expressions by running them sequentially with persistent state.
    
    This replaces vectorized batch scoring for stateful expressions.
    Each call to score() runs the expression for every timestep in order,
    carrying the state dict from one timestep to the next.
    """

    def __init__(self, n_state_vars: int = 4):
        self.n_state_vars = n_state_vars
        self._mdl = MDLEngine()

    def score(
        self,
        expr: ExtExprNode,
        observations: List[int],
        initial_state: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, Dict[int, float]]:
        """
        Score an expression by sequential evaluation with state.
        
        Returns: (mdl_cost, final_state)
        """
        state = copy.deepcopy(initial_state) if initial_state else {}
        predictions = []

        for t, actual in enumerate(observations):
            try:
                pred = expr.evaluate(t, observations[:t], state)
                if not isinstance(pred, (int, float)) or not math.isfinite(pred):
                    pred = 0.0
                predictions.append(int(round(pred)))
            except Exception:
                predictions.append(0)

        try:
            result = self._mdl.compute(
                predictions, observations,
                expr.node_count(), expr.constant_count()
            )
            return result.total_mdl_cost, state
        except Exception:
            return float('inf'), state

    def score_batch_stateful(
        self,
        exprs: List[ExtExprNode],
        observations: List[int],
        initial_state: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[float, Dict[int, float]]]:
        """Score multiple expressions, each with fresh initial state."""
        return [self.score(expr, observations, copy.deepcopy(initial_state)) for expr in exprs]


class StatefulBeamSearch:
    """
    Beam search that handles STATE_VAR nodes correctly.
    
    Uses StatefulScorer instead of batch vectorized scoring.
    Generates expressions that can use STATE_VAR(0..n_state_vars-1).
    Adds STATE_VAR-specific warm-start seeds for algorithm environments.
    """

    def __init__(self, config: StatefulSearchConfig = None):
        self.cfg = config or StatefulSearchConfig()
        self._scorer = StatefulScorer(n_state_vars=self.cfg.n_state_vars)
        self._rng = random.Random(self.cfg.random_seed)

        # Underlying grammar beam for expression generation and mutation
        self._grammar_beam = GrammarConstrainedBeam(GrammarBeamConfig(
            beam_width=self.cfg.beam_width,
            max_depth=self.cfg.max_depth,
            const_range=self.cfg.const_range,
            max_lag=self.cfg.max_lag,
            n_iterations=1,  # we handle iteration externally
            random_seed=self.cfg.random_seed,
        ))

    def _gcd_warm_starts(self, alphabet_size: int) -> List[ExtExprNode]:
        """
        Warm-start seeds for GCDEnv.
        The Euclidean algorithm: GCD(a,b) = GCD(b, a%b) until b=0.
        
        With STATE_VAR nodes:
          STATE(0) = a  (larger number)
          STATE(1) = b  (smaller number)
          output = STATE(0) when STATE(1) = 0
        
        Approximated as: GCD_NODE(obs[t] // encoding, obs[t] % encoding)
        or: MOD(STATE(0), STATE(1)) operating on the input
        """
        from ouroboros.synthesis.expr_node import NodeType, ExprNode

        seeds = []

        # Seed 1: GCD_NODE directly on some encoding of (a, b)
        # For GCDEnv: inputs are encoded as a*21 + b
        # GCD(a, b) = GCD_NODE(input // 21, input % 21)
        def const_e(v):
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.CONST; n.value = float(v)
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n

        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n

        # Seed: GCD_NODE(FLOOR(TIME / 21), MOD(TIME, 21))
        # This decodes the GCDEnv encoding
        encoding = 21  # GCDEnv.ENCODING
        floor_div = ExtExprNode(ExtNodeType.FLOOR_NODE,
            left=ExtExprNode.__new__(ExtExprNode))
        # Build TIME / encoding
        from ouroboros.synthesis.expr_node import NodeType as OldNT
        time_div_enc = ExprNode(OldNT.DIV,
            left=ExprNode(OldNT.TIME),
            right=ExprNode(OldNT.CONST, value=encoding))
        mod_enc = ExprNode(OldNT.MOD,
            left=ExprNode(OldNT.TIME),
            right=ExprNode(OldNT.CONST, value=encoding))

        # Package into ExtExprNode wrappers
        gcd_seed = ExtExprNode(ExtNodeType.GCD_NODE)
        gcd_seed.left = ExtExprNode.__new__(ExtExprNode)
        gcd_seed.left.node_type = ExtNodeType.FLOOR_NODE
        gcd_seed.left.value = 0.0; gcd_seed.left.lag = 1
        gcd_seed.left.state_key = 0; gcd_seed.left.window = 10
        gcd_seed.left.left = const_e(0)  # placeholder
        gcd_seed.left.right = None; gcd_seed.left.third = None; gcd_seed.left._cache = {}

        gcd_seed.right = const_e(float(encoding))
        gcd_seed.third = None; gcd_seed._cache = {}

        seeds.append(gcd_seed)

        # Seed 2: Just GCD_NODE(CONST(a), CONST(b)) for various a,b pairs
        for a in [6, 12, 15]:
            for b in [4, 8, 10]:
                if math.gcd(a, b) > 1:
                    node = ExtExprNode(ExtNodeType.GCD_NODE,
                                       left=const_e(float(a)),
                                       right=const_e(float(b)))
                    seeds.append(node)

        return seeds

    def search(
        self,
        observations: List[int],
        alphabet_size: int,
        env_name: str = "unknown",
        initial_state: Optional[Dict[int, float]] = None,
        verbose: bool = False,
    ) -> Optional[ExtExprNode]:
        """Run stateful beam search."""
        beam_width = self.cfg.beam_width

        # Initialize beam
        seeds = []
        if "GCD" in env_name.upper():
            seeds = self._gcd_warm_starts(alphabet_size)

        # Generate initial candidates
        init_exprs = seeds[:]
        for _ in range(beam_width * 4):
            init_exprs.append(self._grammar_beam._random_expr())

        # Score all initial candidates
        scored = self._scorer.score_batch_stateful(init_exprs, observations, initial_state)
        candidates = sorted(
            [StatefulCandidate(e, c, s) for e, (c, s) in zip(init_exprs, scored)],
        )[:beam_width]

        if verbose:
            print(f"  StatefulBeam initial best: {candidates[0].mdl_cost:.2f}")

        # Iterate
        for iteration in range(self.cfg.n_iterations):
            new_candidates = list(candidates)
            for cand in candidates:
                for _ in range(3):
                    mutated = self._grammar_beam._mutate_grammar(cand.expr)
                    cost, final_state = self._scorer.score(mutated, observations, initial_state)
                    new_candidates.append(StatefulCandidate(mutated, cost, final_state))

            new_candidates.sort()
            candidates = new_candidates[:beam_width]

        if verbose:
            best = candidates[0]
            print(f"  StatefulBeam final: {best.mdl_cost:.2f}")
            print(f"  Expression: {best.expr.to_string()[:60]}")

        return candidates[0].expr if candidates else None


class StatefulHierarchicalRouter:
    """
    Extended HierarchicalSearchRouter that handles STATE_VAR expressions.
    
    When the environment is classified as requiring stateful computation
    (GCDEnv, algorithm environments), uses StatefulBeamSearch.
    Otherwise falls back to the standard HierarchicalSearchRouter.
    """

    def __init__(
        self,
        beam_width: int = 20,
        max_depth: int = 5,
        n_iterations: int = 10,
        n_state_vars: int = 4,
        random_seed: int = 42,
    ):
        from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
        self._standard_router = HierarchicalSearchRouter(RouterConfig(
            beam_width=beam_width,
            max_depth=max_depth,
            n_iterations=n_iterations,
            random_seed=random_seed,
        ))
        self._stateful_search = StatefulBeamSearch(StatefulSearchConfig(
            beam_width=beam_width,
            max_depth=max_depth,
            n_iterations=n_iterations,
            n_state_vars=n_state_vars,
            random_seed=random_seed,
        ))
        self._stateful_scorer = StatefulScorer(n_state_vars=n_state_vars)

    def search(
        self,
        observations: List[int],
        alphabet_size: int,
        env_name: str = "unknown",
        use_stateful: bool = None,
        verbose: bool = False,
    ) -> Tuple[Optional[ExtExprNode], float]:
        """
        Search for the best expression, choosing stateful or standard mode.
        
        Returns: (best_expr, best_mdl_cost)
        """
        # Auto-detect: use stateful if env name suggests algorithm environment
        if use_stateful is None:
            stateful_keywords = ["GCD", "Algorithm", "Collatz", "Prime", "Sort"]
            use_stateful = any(kw.upper() in env_name.upper() for kw in stateful_keywords)

        if use_stateful:
            if verbose:
                print(f"  [StatefulRouter] Using stateful search for {env_name}")
            expr = self._stateful_search.search(
                observations, alphabet_size, env_name,
                initial_state={}, verbose=verbose,
            )
            if expr is None:
                return None, float('inf')
            cost, _ = self._stateful_scorer.score(expr, observations)
            return expr, cost
        else:
            result = self._standard_router.search(observations, alphabet_size, verbose=verbose)
            return result.expr, result.mdl_cost