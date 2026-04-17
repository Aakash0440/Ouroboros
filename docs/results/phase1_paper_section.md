# Phase 1: Mathematical Structure Emergence via MDL Compression
## Draft Paper Section — OUROBOROS

---

### 3. Phase 1: Compression-Driven Mathematical Emergence

#### 3.1 Setup

We place N=8 agents in a shared observation environment that generates
a stream of discrete symbols. Each agent independently searches for the
shortest program that predicts the stream, measured by Minimum Description
Length (MDL) cost:

    L(M, D) = λ · L(M) + L(D | M)

where L(M) is the compressed size of the program M in bytes, L(D|M) is
the bits required to encode prediction errors under M, and λ=1.0 is
the regularization weight (equal weight on model and data).

Agents search over symbolic expression trees with five node types:
CONST(n), TIME(t), ADD, MUL, MOD. Search proceeds via beam search
(width K=25, max depth 3) followed by MCMC refinement (simulated
annealing, 200 iterations).

#### 3.2 Environments

We test on five environments:

| Environment | Alphabet | Hidden Rule |
|-------------|----------|-------------|
| BinaryRepeat | {0,1} | t mod 2 |
| ModularArith(7,3,1) | {0..6} | (3t+1) mod 7 |
| FibonacciMod(11) | {0..10} | F(t) mod 11 |
| PrimeSequence | {0,1} | is_prime(t) |
| Noise | {0..3} | None (random) |

The landmark environment is ModularArith(7,3,1). Agents are given no
prior knowledge of modular arithmetic, the modulus, slope, or intercept.

#### 3.3 Proto-Axiom Pool

After each checkpoint, agents submit their best symbolic expression to
a shared ProtoAxiomPool. The pool computes a behavioral fingerprint
(predictions on t=0..99) for each expression and groups agents by
identical fingerprints. When ≥ 50% of agents independently submit
expressions with the same fingerprint, the expression is promoted to a
proto-axiom with confidence:

    confidence = (support_fraction) × (compression_improvement)

where compression_improvement = 1 - (MDL_cost / naive_bits).

#### 3.4 Results

**Table 1: Phase 1 Compression Results (N=8 agents, 2000 observations)**

[Numbers from experiments/phase1/results/phase1_all_results.json]

**Figure 1: Mathematical Discovery Event**

Figure 1 shows compression ratio vs. observation count for 8 agents
on ModularArith(7,3,1). At step ~400, one agent's ratio drops from
~0.20 to 0.004 — a 50× improvement. This agent had independently
assembled the expression `(t * 3 + 1) mod 7` from the primitives
CONST, TIME, MUL, ADD, MOD without any prior knowledge of modular
arithmetic.

By step 600, 5/8 agents had converged on the same behavioral fingerprint,
triggering promotion of proto-axiom AX_00001 with confidence 0.62.

**Key finding:** Arithmetic structure (modular arithmetic) emerged from
compression pressure alone. The system was never shown the rule, the
modulus, or the concept of modular reduction.

**Noise control:** The Noise environment correctly produced zero proto-axioms,
confirming the pool does not false-positive on random data.

#### 3.5 Multi-Scale Emergence

[Section from multi-scale experiment results]

HierarchicalAgents on MultiScaleEnv(slow=28, fast=7) discovered structure
at TWO independent temporal scales:
- Scale 1 (symbol level): fast-period pattern
- Scale ≥4 (window level): slow-period pattern

Cross-scale consistency scores confirmed these are complementary axioms,
not conflicting ones. This extends the single-scale result to environments
with hierarchical mathematical structure.

#### 3.6 Discussion

The Phase 1 result establishes that:
1. MDL pressure is sufficient to cause mathematical structure discovery
2. Multi-agent consensus is necessary to distinguish real structure from
   overfitting (noise control)
3. Multi-scale compression extends the result to hierarchically structured
   environments

Phase 2 tests whether these proto-axioms survive adversarial attack in
the cryptographic proof market — a more stringent test of their validity.
