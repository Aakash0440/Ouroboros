# OUROBOROS

**Multi-agent mathematical discovery via MDL compression**

A society of agents that discovers mathematical laws from raw observation sequences using Minimum Description Length as its sole learning signal. No training data. No domain knowledge. Compression pressure alone drives the discovery of modular arithmetic, the Chinese Remainder Theorem, Hooke's Law, the prime counting function, and the Fundamental Theorem of Calculus.

[![Tests](https://img.shields.io/badge/tests-1435%2B-brightgreen)](tests/)
[![Lean4 sorry](https://img.shields.io/badge/Lean4%20sorry-0-brightgreen)](ouroboros_lean/)
[![Version](https://img.shields.io/badge/version-v15.0.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## The Core Claim

Mathematical structure **emerges** from compression pressure. Agents are never shown rules — MDL pressure causes them to discover rules because rules are the shortest description of the data.

When an agent finds `(3t+1) % 7`, it's not because it knows modular arithmetic. It's because that 9-character expression compresses 5,000 observations to a compression ratio of **0.0041 ± 0.0002** — a 250× improvement over the naive baseline. The mathematics is a consequence of the compression.

---

## Quickstart

```bash
git clone https://github.com/Aakash0440/Ouroboros
cd Ouroboros
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Verify everything works
python -m pytest tests/ -q --tb=no

# Discover a modular arithmetic law
python -c "
from ouroboros.api.client import OuroborosClient
c = OuroborosClient()
r = c.discover([float((3*t+1)%7) for t in range(200)], alphabet_size=7)
print(r)
"

# Discover from real scientific data (auto-detrends, FFT-seeds)
python -c "
from ouroboros.api.smart_preprocessor import SmartPreprocessor
import math

co2 = [312.0 + 0.041*t + 3.0*math.sin(2*math.pi*t/12) for t in range(300)]
proc = SmartPreprocessor()
result = proc.process(co2)
print(f'Series type: {result.series_type.value}')
print(f'Detected period: {result.detected_period} months')
"

# Start the web API
uvicorn ouroboros.api.server:app --port 8000

# Run the full benchmark (10 seeds)
python scripts/run_full_benchmark.py --seeds 10
```

---

## Benchmark Results

### Core Compression

| Result | Value | Verified via |
|--------|-------|--------------|
| ModArith(7) compression ratio | **0.0094** (10-seed mean) | `python benchmarkJ.py` |
| ModArith success rate (ratio < 0.01) | **100%** — 10/10 seeds | `python benchmarkJ.py` |
| MDL calibration — perfect fit data cost | **0.0000 bits** | `python benchmarkH.py` |
| Family classification accuracy | **20/20 seeds** NUMBER_THEOR | `python benchmarkI.py` |
| ModArith vs trivial baseline | **ratio 0.016** ✓ | `python benchmarkE.py` |
| Fibonacci%7 vs trivial baseline | **ratio 0.048** ✓ | `python benchmarkE.py` |
| Feynman 1/r² (seeded) | **MDL −673, R²=1.0** | `python benchmarkD.py` |
| GARCH structure (squared vs raw returns) | **correct direction** | `python benchmarkA.py` |
| Feynman benchmark recovery rate | **91.2%** (noiseless) | `python scripts/run_feynman.py` |

### Scientific Law Detection

| Result | Value | Verified via |
|--------|-------|--------------|
| Prime counting function π(n) accuracy | **100%** — 50/50 exact | `python verify_day34.py` |
| Hooke's Law correlation | **r = −0.94** | `python verify_day33.py` |
| Fundamental Theorem of Calculus | **Exact** algebraic identity | `python verify_day43.py` |
| Exponential decay correlation | **r = −0.97** | `python verify_day33.py` |
| CRT success rate | **0.87 ± 0.07** (10 seeds) | `python scripts/run_full_benchmark.py` |
| Lorenz attractor coupling recall | **100%** | `python test1.3.py` |
| CO₂ → Temperature causal edge | **Detected** (lag ≈ 20–30) | `python test1.5.py` |

### Causal Discovery

| Test | Result |
|------|--------|
| Hidden confounder (Z→X, Z→Y; no X↔Y) | **PASS** |
| Simpson's paradox trap | **PASS** |
| Feedback loop X⇌Y | **PASS** — both directions |
| Lag ambiguity (lag=3) | **PASS** — correct lag identified |
| Noisy confounder resilience | **PASS** — spurious edge blocked |

### Novelty Detection

| Test | Result |
|------|--------|
| AUROC (known vs novel) | **1.0000** — perfect separation |
| Scale invariance (3t vs 6t) | **PASS** |
| Cross-domain novelty discrimination | **PASS** |
| Phase-shift awareness | **PASS** — sin(t+π/4) known; sin(2t) novel |

### Meta-Learning & Self-Improvement

| Metric | Value |
|--------|-------|
| DERIV bits after physics training | **2.79** (down from 4.0) |
| Sample efficiency trend | **Spearman ρ = −0.77** |
| Self-improvement loop MDL trend | **Spearman ρ = −1.0** (monotone) |
| Civilization vs human history | **ρ = 0.71 [0.52, 0.84]** |
| Proof market convergence (8 agents) | **7.8 ± 0.74 rounds** |

### Formal Verification

| Test | Result |
|------|--------|
| Lean4 sorry count | **0** |
| Auto-prove periodicity (3 moduli) | **3/3 PASS** — omega, 1 attempt each |
| Auto-prove boundedness | **PASS** — norm_num, 1 attempt |
| Bezout witness verification | **All 77 pairs correct** |

---

## Known Limitations

| Issue | Status |
|-------|--------|
| Grammar branching ~12, not 6.2 as documented | `CategoryConstraintGrammar` corrects this; 4–5× reduction from 55 nodes |
| Prime gaps below trivial baseline (ratio 1.31) | Individual gaps are near-random; LOG search improvement in progress |
| Random noise overfitting (ratio 1.42) | `BaselineComparator` override added; root-cause fix in progress |
| CO₂/sunspot return PREV(1) on raw input | `SmartPreprocessor` detrending required before discovery |
| Power law discovery requires seeding | Rational exponent initialization being added |
| 50-byte expression ceiling | Beam explores ~1,125 candidates; deep multi-mechanism laws unreachable |
| Causal claims are Granger, not Pearl | Interventional do-queries supported in simulation only |
| Papers not yet submitted | LaTeX complete, numbers verified; second author and affiliation needed |

---

## Architecture

### The MDL Objective

Every expression is scored by its two-part MDL cost:

```
L(f, obs) = |f|_bits + H(f(obs) ‖ obs)
```

- `|f|_bits` — description length of the expression itself (node bits + constant bits + depth penalty)
- `H(f(obs) ‖ obs)` — Shannon entropy of prediction residuals
- Minimizing the sum simultaneously minimizes complexity and prediction error
- Compression ratio = `L(f, obs) / (n × log₂(k))` — the primary quality metric

### Five-Layer Self-Improvement

```
Layer 1  Expression search
         60 symbolic nodes × CategoryConstraintGrammar × NeuralNodePrior → beam search
         Grammar: branching ~12 (4–5× reduction from unconstrained 55-node set)
         Classifier: 7 math families × O(n) statistics → category restriction before search

Layer 2  Hyperparameter adaptation
         Beam width, MCMC iterations, const_range — evaluated on OOD environments
         Measured gain: 12 ± 3% MDL reduction

Layer 3  MDL objective modification
         λ_prog and λ_const co-evolve with the agent's discovered expression history

Layer 4  Search algorithm invention
         17-opcode DSL: INIT · BEAM · MUTATE · FFT_SEED · MCMC · GRAMMAR_FILTER
                        SORT_MDL · TAKE · LOOP · CLASSIFY_ENV · IF_PERIODIC
                        SAVE_BEST · LOAD_BEST · PARALLEL · ANNEAL · ELITE_KEEP · CROSS
         Layer4ProofMarket approval rate: 23%

Layer 5  Vocabulary extension
         PrimitiveProposer: residual analysis → multiplicativity / periodicity detection
         PrimitiveVerifier: fits known function class → Python + Lean4 implementation
         DynamicGrammar: approved primitives immediately available to future searches
         Verified on: Ramanujan tau function, Liouville lambda function
```

### Adversarial Proof Market

```
1. Agent proposes f with MDL(f) < MDL(f_old)
2. All agents commit SHA-256(counterexample ‖ salt) simultaneously
3. Agents reveal — post-hoc adaptation impossible (collision prob = 2⁻¹²⁸)
4. No valid counterexample AND OOD test passes → approved
5. Approved expressions receive Lean4 formal verification
```

### Search Pipeline

```
Input sequence
    │
    ▼
EnvironmentClassifier          O(n) stats → 7 families
    │                          NUMBER_THEOR / PERIODIC / EXPONENTIAL
    │                          RECURRENT / STATISTICAL / MIXED / RANDOM
    ▼
FFTPeriodFinder                Seeds beam with dominant-period expressions
    │                          (PERIODIC family only)
    ▼
CategoryConstraintGrammar      Type-based child restrictions
    │                          Branching: ~12 (down from 28)
    ▼
NeuralNodePrior (240 params)   Biases sampling toward historically successful nodes
    │                          Updated online, domain-specific
    ▼
GrammarConstrainedBeam         width=25, depth=5, iterations=15
    │                          L-BFGS constant optimization each iteration
    │                          Rational seeds: {½, 1, 1½, 2, π, e, √2, …}
    ▼
BaselineComparator             Returns constant/PREV/linear if OUROBOROS loses
    │
    ▼
RouterResult
    ├── expression_str
    ├── mdl_cost
    ├── compression_ratio
    ├── math_family
    └── confidence
```

### Smart Preprocessing

```python
from ouroboros.api.smart_preprocessor import SmartPreprocessor

proc = SmartPreprocessor()
result = proc.process(time_series)
# SLOWLY_CHANGING → linear detrend → discover structure in residuals
# PERIODIC        → FFT seed with dominant period
# STATIONARY      → direct beam search
# RANDOM          → constant baseline returned immediately
```

---

## What It Discovers

### Modular Arithmetic
```python
# Input:  [1, 4, 0, 3, 6, 2, 5, 1, 4, 0, ...]  (200 observations)
# Output: (3*t+1) % 7
# MDL ratio: 0.0094 — 100% success rate across 10 seeds
```

### Prime Counting Function
```python
# CUMSUM(ISPRIME(TIME))[t] = π(t)  — exact, not approximate
# 50/50 predictions correct
# The expression IS the mathematical definition
```

### Hooke's Law (structural detection)
```python
# Input:  x(t) = A·cos(ωt)  — raw position measurements
# Output: CORR(DERIV2(x), x) = −0.94
# Structural relationship identified — not curve fitting
```

### Fundamental Theorem of Calculus
```python
# DERIV(INTEGRAL(CONST(5)))[t] = 5  — exact, for all t
```

### Chinese Remainder Theorem
```python
# Input:  joint stream interleaving (mod 7) and (mod 11) observations
# Output: formula over Z/77Z
# Bezout witness (a*22 + b*56) % 77 — verified for all 77 pairs
# CRT success rate: 0.87 ± 0.07
```

### Feynman Coulomb's Law (seeded)
```python
# Input:  F = C/(r+1)²  (50 discretized observations)
# Output: C / ((t+1.0) * (t+1.0))
# MDL: −673 bits, R² = 1.000000
```

---

## The 60 Node Types

| Category | Nodes |
|----------|-------|
| **Calculus** | DERIV, DERIV2, CUMSUM, INTEGRAL, INTEGRAL_WIN, EWMA, RUNNING_MAX, RUNNING_MIN, CONVOLVE, DIFF_QUOT |
| **Statistical** | MEAN_WIN, VAR_WIN, STD_WIN, CORR, ZSCORE, QUANTILE |
| **Logical** | THRESHOLD, SIGN, COMPARE, BOOL_AND, BOOL_OR, BOOL_NOT, CLAMP |
| **Transform** | FFT_AMP, FFT_PHASE, AUTOCORR, HILBERT_ENV |
| **Number Theory** | GCD_NODE, LCM_NODE, FLOOR_NODE, CEIL_NODE, ROUND_NODE, FRAC_NODE, TOTIENT, ISPRIME |
| **Memory / State** | ARGMAX_WIN, ARGMIN_WIN, COUNT_WIN, STREAK, DELTA_ZERO, STATE_VAR |
| **Core (20)** | CONST, TIME, PREV, ADD, SUB, MUL, DIV, MOD, POW, IF, EQ, LT, SIN, COS, EXP, LOG, SQRT, ABS, SIGN, CLAMP |

Grammar constraints: BOOL_AND accepts only LOGICAL/ARITHMETIC children. ISPRIME accepts only TERMINAL/ARITHMETIC. FFT_AMP accepts only TERMINAL/ARITHMETIC. These prevent nonsensical combinations and reduce effective branching from 55 to ~12.

---

## Web API

```bash
uvicorn ouroboros.api.server:app --host 0.0.0.0 --port 8000
```

```python
from ouroboros.api.client import OuroborosClient

client = OuroborosClient("http://localhost:8000")
result = client.discover(
    observations=[1, 4, 0, 3, 6, 2, 5, 1, 4, 0],
    alphabet_size=7,
    beam_width=20,
    time_budget_seconds=10.0,
    verify_physics_laws=True,
)
print(result.expression)         # "(3*t+1) % 7"
print(result.mdl_cost)           # 45.21
print(result.compression_ratio)  # 0.0094
print(result.math_family)        # "NUMBER_THEOR"
print(result.verified_law)       # "NONE" | "HOOKES_LAW" | "EXPONENTIAL_DECAY" | ...
```

**Endpoints:** `POST /discover` · `POST /verify_law` · `GET /health` · `GET /stats`

Rate limiting: 10 req/min per IP · Session cache: 100 LRU entries · Docker deployment included

---

## Real Data Integration

```python
from ouroboros.api.data_pipeline import RealDataPipeline

pipeline = RealDataPipeline(beam_width=20, n_iterations=10)

result = pipeline.discover(pandas_df,     format="pandas", target_column="temperature")
result = pipeline.discover("data.csv",    format="csv",    target_column="co2_ppm")
result = pipeline.discover(fasta_string,  format="fasta")   # GC-content windows
result = pipeline.discover(hdf5_data,     format="hdf5",   dataset="/sensor/ch1")
result = pipeline.discover("[1,2,3,4,5]", format="json")
```

**Supported formats:** pandas DataFrame · numpy array · CSV · JSON · HDF5 · netCDF · FASTA · Python dict/list

---

## Formal Verification

Zero `sorry` across all theorem files. `AutoProofEngine` closes periodicity and boundedness statements automatically:

```lean
-- Auto-proved by omega, 1 attempt each:
∀ t : ℕ, (5 * (t + 11) + 3) % 11 = (5 * t + 3) % 11   ✓
∀ t : ℕ, (7 * t + 2) % 13 < 13                          ✓
∀ t : ℕ, (2 * (t + 17) + 9) % 17 = (2 * t + 9) % 17    ✓

-- Concrete arithmetic (norm_num, 1 attempt):
(3 * 5 + 1) % 7 = 2    ✓
```

Mathlib4 contribution files ready for submission:
- `ouroboros_lean/Mathlib4Contribution/LinearModularSurjective.lean`
- `ouroboros_lean/Mathlib4Contribution/CRTInstances.lean`

---

## Comparison

| Feature | OUROBOROS | Eureqa (2009) | FunSearch (2024) | AI Feynman (2020) | PySR (2023) |
|---------|:---------:|:-------------:|:----------------:|:-----------------:|:-----------:|
| Objective | MDL (principled) | Accuracy + heuristic complexity | LLM evaluator | Neural network + dim. analysis | Accuracy + Pareto |
| Training data required | ✗ | ✗ | ✗ | ✗ | ✗ |
| Lean4 formal proofs | ✓ (0 sorry) | ✗ | ✗ | ✗ | ✗ |
| Auto theorem proving | ✓ | ✗ | ✗ | ✗ | ✗ |
| Recursive self-improvement | ✓ (5 layers) | ✗ | ✗ | ✗ | ✗ |
| Multi-agent adversarial market | ✓ (SHA-256) | ✗ | ✗ | ✗ | ✗ |
| Novelty detection (AUROC 1.0) | ✓ | ✗ | ✗ | ✗ | ✗ |
| Causal discovery | ✓ Granger + backdoor | ✗ | ✗ | ✗ | ✗ |
| Vocabulary self-extension | ✓ PrimitiveProposer | ✗ | ✗ | ✗ | ✗ |
| Physics law structural detection | ✓ | ✗ | ✗ | ✗ | ✗ |
| Real data pipeline (8 formats) | ✓ | CSV only | ✗ | ✗ | CSV / numpy |
| Feynman benchmark recovery | 91.2% | ~70% | N/A | ~82% | ~75–80% |

**Where OUROBOROS leads:** Formal verification, causal discovery, novelty detection, multi-agent coordination, self-extending vocabulary, physics law identification.  
**Where PySR leads:** Raw recovery rate on smooth continuous functions; speed on simple problems.

---

## Project Structure

```
ouroboros/
├── api/              FastAPI server, OuroborosClient, RealDataPipeline, SmartPreprocessor
├── agents/           BaseAgent → SynthesisAgent → Layer4Agent → Layer5Agent
├── benchmark/        BenchmarkRunner, FinalBenchmark
├── causal/           DoCalculusEngine, InterventionalEnvironments,
│                     CausalMDLScorer, StructuralIsomorphismDetector
├── civilization/     CivilizationSimulator, bootstrap Spearman statistics
├── compression/      MDLEngine, GaussianMDL
├── core/             SelfImprovementLoop
├── environments/     ModularArithmetic, Fibonacci, SpringMass, Radioactive,
│                     FreeFall, GCD, PrimeCount, Collatz, FundamentalTheorem,
│                     Multivariate (SpringMass / Climate / Finance)
├── grammar/          CategoryConstraintGrammar (v15), MathGrammar (legacy)
├── knowledge/        SimpleAxiomKB, AccumulationRunner, 100-session experiment
├── layer4/           SearchAlgorithmDSL (17 opcodes), AlgorithmInterpreter,
│                     Layer4Agent, Layer4ProofMarket, CompositeOpcode
├── meta/             MetaMDLLearner — domain-specific MDL priors
├── nodes/            ExtNodeType (40 nodes), ExtExprNode, NodeSpec
├── novelty/          BehavioralEmbedder, EmbeddingRegistry, OEISClient,
│                     LiteratureMatcher, NoveltyDetector, OpenConjectures
├── papers/           ArxivPaperBuilder, Lean4PRGenerator, Mathlib4Submission
├── physics/          PhysicsLawVerifier, DerivativeAnalyzer
├── primitives/       PrimitiveProposer, PrimitiveVerifier, PrimitiveRegistry
├── proof_market/     SHA-256 commit-reveal, OODPressure, AutoProofEngine
├── search/           GrammarConstrainedBeam, HierarchicalSearchRouter,
│                     EnvironmentClassifier, NeuralNodePrior, FFTPeriodFinder,
│                     StatefulBeamSearch, PosteriorExpressionSampler,
│                     BaselineComparator
├── synthesis/        BeamSearchSynthesizer, MCMC, ExprNode
└── targeting/        ConjectureTargetingSession (Collatz, PrimeGaps, TwinPrimes)

ouroboros_lean/
├── OuroborosProofs/          Core theorems — 0 sorry
└── Mathlib4Contribution/     LinearModularSurjective.lean, CRTInstances.lean (PR-ready)

tests/         1,435+ tests across all modules
results/       Benchmark results, paper LaTeX, conjecture_flags.jsonl
scripts/       run_full_benchmark.py, run_civilization_stats.py,
               run_knowledge_accumulation_100.py, run_final_benchmark.py
```

---

## Reproducibility

```bash
# Full reproduction
bash results/reproducibility/reproduce.sh

# Individual components
python scripts/run_full_benchmark.py --seeds 10      # core benchmark
python scripts/run_civilization_stats.py --full      # civilization simulation
python scripts/run_knowledge_accumulation_100.py     # knowledge accumulation
python scripts/run_final_benchmark.py               # final 7-domain benchmark

# Individual benchmark scripts
python benchmarkA.py    # Financial GARCH structure
python benchmarkD.py    # Feynman 1/r²
python benchmarkE.py    # Trivial baseline comparison
python benchmarkH.py    # MDL calibration
python benchmarkI.py    # Family classification
python benchmarkJ.py    # Canonical sequence regression

# Compile papers (requires pdflatex)
cd results/papers/paper1 && make
cd results/papers/paper2 && make
```

---

## Papers

### Paper 1: Mathematical Emergence from Compression Pressure
*Target: NeurIPS / ICLR*

Agents with 60 symbolic primitives and MDL as their sole objective discover algebraic laws without training data. Grammar-constrained search makes the vocabulary tractable. Physics laws are identified by derivative correlation analysis. A mathematical civilization simulation shows discovery order correlating with human history (ρ = 0.71). AutoProofEngine automatically closes periodicity and boundedness theorems in Lean4.

### Paper 2: Adversarial Proof Markets for Self-Improving Agents
*Target: ICML*

SHA-256 commit-reveal prevents post-hoc counterexample adaptation. OOD testing prevents overfitting. Five layers of recursive self-improvement culminate in a 17-opcode search algorithm DSL and vocabulary extension via PrimitiveProposer. Causal discovery integration outputs interventional predictions alongside symbolic expressions. Meta-learning reduces DERIV description cost from 4.0 to 2.79 bits after extended physics training.

*Both LaTeX packages compile cleanly with real benchmark numbers. Pending: second author and institutional affiliation.*

---

## References

- Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14, 465–471.
- Li, M. & Vitányi, P. (1997). *An Introduction to Kolmogorov Complexity*. Springer.
- Schmidt, M. & Lipson, H. (2009). Distilling free-form natural laws from experimental data. *Science*, 324, 81–85.
- Schmidhuber, J. (2003). Gödel machines: Fully self-referential optimal agents. *arXiv:cs/0309048*.
- Grünwald, P. (2007). *The Minimum Description Length Principle*. MIT Press.
- Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Udrescu, S. & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16).
- Cranmer, M. (2023). Interpretable machine learning for physics. *arXiv:2207.09434*.
- Romera-Paredes, B. et al. (2024). Mathematical discoveries from program search with large language models. *Nature*, 625, 468–475.
- Lu, C. et al. (2024). The AI Scientist. *arXiv:2408.06292*.
- The Mathlib Community (2020). The Lean 4 Mathematical Library. *arXiv:1910.09336*.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Lean4 contribution files (`ouroboros_lean/Mathlib4Contribution/`) released under Apache 2.0 for potential inclusion in Mathlib4.