"""
AlgorithmInterpreter — Runs a DSL search algorithm program.

The interpreter maintains an execution context:
  beam: current list of (expression, mdl_cost) candidates
  best: saved best expression
  is_periodic: result of the last CLASSIFY_ENV call
  seed_pool: extra expressions from FFT or classification

And executes each DSL instruction against real observations.
"""

from __future__ import annotations
import copy
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from ouroboros.layer4.search_dsl import (
    DSLInstruction, DSLOpcode, SearchAlgorithmProgram
)
from ouroboros.nodes.extended_nodes import ExtExprNode
from ouroboros.grammar.math_grammar import DEFAULT_GRAMMAR
from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
from ouroboros.compression.mdl_engine import MDLEngine


@dataclass
class InterpreterContext:
    """Execution context for the DSL interpreter."""
    observations: List[int]
    alphabet_size: int
    beam: List[Tuple[ExtExprNode, float]]   # (expr, mdl_cost)
    saved_best: Optional[Tuple[ExtExprNode, float]] = None
    is_periodic: bool = False
    seed_pool: List[ExtExprNode] = field(default_factory=list)
    category_weights: Dict = field(default_factory=dict)
    random_seed: int = 42
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        self._rng = random.Random(self.random_seed)

    @property
    def best_expr(self) -> Optional[ExtExprNode]:
        if self.beam:
            return self.beam[0][0]
        if self.saved_best:
            return self.saved_best[0]
        return None

    @property
    def best_cost(self) -> float:
        if self.beam:
            return self.beam[0][1]
        if self.saved_best:
            return self.saved_best[1]
        return float('inf')


class AlgorithmInterpreter:
    """
    Runs a SearchAlgorithmProgram against observations.
    Returns the best expression found and its MDL cost.
    """

    def __init__(self, time_budget_seconds: float = 10.0):
        self.time_budget = time_budget_seconds
        self._mdl = MDLEngine()
        self._grammar_beam_base = GrammarConstrainedBeam(
            GrammarBeamConfig(beam_width=15, max_depth=4, n_iterations=3)
        )

    def run(
        self,
        program: SearchAlgorithmProgram,
        observations: List[int],
        alphabet_size: int,
        verbose: bool = False,
    ) -> Tuple[Optional[ExtExprNode], float, float]:
        """
        Execute the program against observations.
        
        Returns: (best_expr, best_mdl_cost, wall_time_seconds)
        """
        start = time.time()

        ctx = InterpreterContext(
            observations=observations,
            alphabet_size=alphabet_size,
            beam=[],
            random_seed=42,
        )

        if verbose:
            print(f"\n[Interpreter] Running: {program.to_string()[:80]}")

        for instr in program.instructions:
            if time.time() - start > self.time_budget:
                break
            self._execute(instr, ctx, start, verbose)

        elapsed = time.time() - start
        return ctx.best_expr, ctx.best_cost, elapsed

    def _score(self, expr: ExtExprNode, ctx: InterpreterContext) -> float:
        """Score an expression under MDL."""
        import math
        try:
            preds = []
            state = {}
            for t in range(len(ctx.observations)):
                p = expr.evaluate(t, ctx.observations[:t], state)
                preds.append(int(round(p)) if math.isfinite(p) else 0)
            r = self._mdl.compute(
                preds, ctx.observations,
                expr.node_count(), expr.constant_count()
            )
            return r.total_mdl_cost
        except Exception:
            return float('inf')

    def _execute(
        self,
        instr: DSLInstruction,
        ctx: InterpreterContext,
        start_time: float,
        verbose: bool,
    ) -> None:
        """Execute one DSL instruction, modifying ctx in place."""
        op = instr.opcode

        if op == DSLOpcode.INIT:
            n = max(5, instr.param)
            beam_cfg = GrammarBeamConfig(
                beam_width=n,
                max_depth=4,
                n_iterations=1,
                random_seed=ctx.random_seed,
            )
            searcher = GrammarConstrainedBeam(beam_cfg)
            # Generate n random grammar-valid expressions
            new_exprs = [searcher._random_expr() for _ in range(n)]
            # Add seed pool
            new_exprs.extend(ctx.seed_pool[:5])
            new_candidates = []
            for e in new_exprs:
                if time.time() - start_time > self.time_budget:
                    break
                new_candidates.append((e, self._score(e, ctx)))
            ctx.beam.extend(new_candidates)
            ctx.beam.sort(key=lambda x: x[1])
            if verbose:
                print(f"  INIT({n}): beam size={len(ctx.beam)}")

        elif op == DSLOpcode.BEAM:
            # Set target beam width (just a hint, TAKE actually truncates)
            pass

        elif op == DSLOpcode.MUTATE:
            n_mutations = max(1, instr.param)
            searcher = GrammarConstrainedBeam(
                GrammarBeamConfig(beam_width=10, random_seed=ctx.random_seed)
            )
            new_candidates = []
            for expr, cost in ctx.beam:
                if time.time() - start_time > self.time_budget:
                    break
                for _ in range(n_mutations):
                    mutated = searcher._mutate_grammar(expr)
                    new_candidates.append((mutated, self._score(mutated, ctx)))
            ctx.beam.extend(new_candidates)
            ctx.beam.sort(key=lambda x: x[1])

        elif op == DSLOpcode.FFT_SEED:
            from ouroboros.search.fft_period_finder import PeriodAwareSeedBuilder
            builder = PeriodAwareSeedBuilder()
            seeds = builder.build_seeds([float(v) for v in ctx.observations])
            ctx.seed_pool.extend(seeds)
            # Also add to beam directly
            for seed in seeds:
                ctx.beam.append((seed, self._score(seed, ctx)))
            ctx.beam.sort(key=lambda x: x[1])

        elif op == DSLOpcode.MCMC:
            # Apply MCMC-style refinement to top-k candidates
            k = min(5, len(ctx.beam))
            searcher = GrammarConstrainedBeam(
                GrammarBeamConfig(beam_width=k, random_seed=ctx.random_seed)
            )
            new_beam = []
            for expr, cost in ctx.beam[:k]:
                for _ in range(max(1, instr.param // 10)):
                    mutated = searcher._mutate_grammar(expr)
                    new_cost = self._score(mutated, ctx)
                    if new_cost < cost:
                        expr, cost = mutated, new_cost
                new_beam.append((expr, cost))
            ctx.beam[:k] = new_beam
            ctx.beam.sort(key=lambda x: x[1])

        elif op == DSLOpcode.GRAMMAR_FILTER:
            # All expressions generated by GrammarConstrainedBeam are already valid
            # This is a no-op in practice but useful conceptually
            ctx.beam = [(e, c) for e, c in ctx.beam if c < float('inf')]

        elif op == DSLOpcode.SORT_MDL:
            ctx.beam.sort(key=lambda x: x[1])

        elif op == DSLOpcode.TAKE:
            k = max(1, instr.param)
            ctx.beam = ctx.beam[:k]

        elif op == DSLOpcode.LOOP:
            n_iters = max(1, instr.param)
            for i in range(n_iters):
                if time.time() - start_time > self.time_budget:
                    break
                for sub_instr in instr.body_a:
                    if time.time() - start_time > self.time_budget:
                        break
                    self._execute(sub_instr, ctx, start_time, verbose)

        elif op == DSLOpcode.CLASSIFY_ENV:
            from ouroboros.search.env_classifier import EnvironmentClassifier
            clf = EnvironmentClassifier()
            result = clf.classify([float(v) for v in ctx.observations])
            ctx.is_periodic = result.primary_family.name == "PERIODIC"
            ctx.category_weights = {
                cat: 2.0 if cat in result.recommended_categories else 0.2
                for cat in result.recommended_categories.__class__.__mro__[0].__subclasses__()
            }

        elif op == DSLOpcode.IF_PERIODIC:
            branch = instr.body_a if ctx.is_periodic else instr.body_b
            for sub_instr in branch:
                self._execute(sub_instr, ctx, start_time, verbose)

        elif op == DSLOpcode.SAVE_BEST:
            if ctx.beam:
                ctx.saved_best = ctx.beam[0]

        elif op == DSLOpcode.LOAD_BEST:
            if ctx.saved_best:
                ctx.beam.append(ctx.saved_best)
                ctx.beam.sort(key=lambda x: x[1])

        elif op == DSLOpcode.PARALLEL:
            k = max(2, instr.param)
            all_beams = []
            for run_i in range(k):
                sub_ctx = InterpreterContext(
                    observations=ctx.observations,
                    alphabet_size=ctx.alphabet_size,
                    beam=[],
                    random_seed=ctx.random_seed + run_i * 17,
                )
                for sub_instr in instr.body_a:
                    self._execute(sub_instr, sub_ctx, start_time, False)
                all_beams.extend(sub_ctx.beam)
            all_beams.sort(key=lambda x: x[1])
            ctx.beam = all_beams



# ── New opcode handlers added Day 42 ──────────────────────────────────────────

def _execute_anneal(
    self,
    instr: 'DSLInstruction',
    ctx: 'InterpreterContext',
    start_time: float,
    verbose: bool,
) -> None:
    """
    ANNEAL(steps) — Simulated annealing on current beam members.
    
    Takes the top beam member, runs annealing from that starting point.
    Temperature schedule: linear decay from 5.0 to 0.1 over `steps` steps.
    """
    import math as _math

    n_steps = max(10, instr.param)
    if not ctx.beam:
        return

    # Start from best beam member
    current_expr, current_cost = ctx.beam[0]
    best_expr, best_cost = current_expr, current_cost
    T_start, T_end = 5.0, 0.1

    searcher = GrammarConstrainedBeam(
        GrammarBeamConfig(beam_width=5, random_seed=ctx.random_seed)
    )

    for step in range(n_steps):
        if time.time() - start_time > self.time_budget:
            break

        T = T_start * ((T_end / T_start) ** (step / max(n_steps - 1, 1)))
        mutated = searcher._mutate_grammar(current_expr)
        mut_cost = self._score(mutated, ctx)

        delta = mut_cost - current_cost
        if delta < 0 or (T > 0 and ctx._rng.random() < _math.exp(-delta / T)):
            current_expr, current_cost = mutated, mut_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_expr = current_expr

    # Add annealing result to beam
    ctx.beam.append((best_expr, best_cost))
    ctx.beam.sort(key=lambda x: x[1])


def _execute_elite_keep(
    self,
    instr: 'DSLInstruction',
    ctx: 'InterpreterContext',
    start_time: float,
    verbose: bool,
) -> None:
    """
    ELITE_KEEP(k) — Save top-k to saved_best, restore after operations.
    
    Ensures the top-k expressions are never lost during restarts.
    Works by updating saved_best to be a list (if k>1) or single best.
    """
    k = max(1, instr.param)
    if not ctx.beam:
        return

    # Save the top-k to a special elite pool in the context
    if not hasattr(ctx, '_elite_pool'):
        ctx._elite_pool = []

    # Merge current beam with elite pool, keep top-k unique
    merged = list(ctx.beam) + ctx._elite_pool
    seen_costs = set()
    unique = []
    for expr, cost in sorted(merged, key=lambda x: x[1]):
        cost_rounded = round(cost, 2)
        if cost_rounded not in seen_costs:
            seen_costs.add(cost_rounded)
            unique.append((expr, cost))
        if len(unique) >= k:
            break
    ctx._elite_pool = unique

    # Also update saved_best with the overall best
    if ctx._elite_pool:
        ctx.saved_best = ctx._elite_pool[0]

    if verbose:
        print(f"  ELITE_KEEP({k}): pool size={len(ctx._elite_pool)}, "
              f"best_cost={ctx._elite_pool[0][1]:.2f}")


def _execute_cross(
    self,
    instr: 'DSLInstruction',
    ctx: 'InterpreterContext',
    start_time: float,
    verbose: bool,
) -> None:
    """
    CROSS(n) — Create n offspring by crossing over pairs of beam members.
    
    Crossover: take the left subtree of parent A, right subtree of parent B.
    This creates expressions that combine the structure of two survivors.
    Better than pure mutation for escaping flat landscapes.
    """
    import copy as _copy, random as _random

    n_cross = max(1, instr.param)
    if len(ctx.beam) < 2:
        return

    offspring = []
    for _ in range(n_cross):
        if time.time() - start_time > self.time_budget:
            break

        # Pick two parents from beam
        parent_a, cost_a = ctx._rng.choice(ctx.beam[:max(2, len(ctx.beam) // 2)])
        parent_b, cost_b = ctx._rng.choice(ctx.beam[:max(2, len(ctx.beam) // 2)])

        if parent_a is parent_b:
            continue

        # Crossover: swap a random subtree
        child = _copy.deepcopy(parent_a)
        donor = _copy.deepcopy(parent_b)

        # Simple crossover: replace left child of root with donor's left child
        if hasattr(child, 'left') and child.left is not None and \
                hasattr(donor, 'left') and donor.left is not None:
            child.left = donor.left

        child_cost = self._score(child, ctx)
        offspring.append((child, child_cost))

    ctx.beam.extend(offspring)
    ctx.beam.sort(key=lambda x: x[1])
    if verbose:
        print(f"  CROSS({n_cross}): added {len(offspring)} offspring")


# Monkey-patch the new handlers into AlgorithmInterpreter
def _execute_extended(self, instr, ctx, start_time, verbose):
    """Extended execute handler for new opcodes."""
    from ouroboros.layer4.search_dsl import DSLOpcode
    op = instr.opcode

    if op == DSLOpcode.ANNEAL:
        _execute_anneal(self, instr, ctx, start_time, verbose)
    elif op == DSLOpcode.ELITE_KEEP:
        _execute_elite_keep(self, instr, ctx, start_time, verbose)
    elif op == DSLOpcode.CROSS:
        _execute_cross(self, instr, ctx, start_time, verbose)
    else:
        # Call original handler
        _original_execute(self, instr, ctx, start_time, verbose)


# Patch the execute method
_original_execute = AlgorithmInterpreter._execute
AlgorithmInterpreter._execute = _execute_extended