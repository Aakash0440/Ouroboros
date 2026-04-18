# OUROBOROS

**Self-bootstrapping mathematical civilization via MDL compression**

A society of agents that, under compression pressure alone, discovers
mathematical structure, verifies self-modifications through a
cryptographic proof market, and derives the Chinese Remainder Theorem
without being told what it is.

---

## The Core Claim

Mathematical structure (modular arithmetic, the Chinese Remainder Theorem)
**emerges** from compression pressure. Agents are never shown the rules.
MDL pressure causes them to discover the rules because the rules are the
shortest description of the data.

---

## Quickstart

```bash
git clone <repo>
cd ouroboros_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Full pipeline: ~30 minutes
python scripts/run_full_pipeline.py

# Quick test: ~3 minutes
python scripts/run_full_pipeline.py --quick

# Individual phases
python experiments/phase1/landmark_experiment.py    # Figure 1
python experiments/phase2/self_modification_experiment.py
python experiments/phase3/crt_landmark_experiment.py  # The CRT result
```

---

## Architecture
Phase 1: MDL Compression
ObservationEnvironment → SynthesisAgent (BeamSearch + MCMC)
→ ProtoAxiomPool (consensus) → Proto-Axiom AX_00001
Phase 2: Proof Market
SelfModifyingAgent → ProofMarket (commit-reveal) → OODPressure
→ Approved modifications → Convergence in ~8 rounds
Phase 3: Causal Theory
TheoryAgent (multi-scale) → CausalTheory → JointEnvironment
→ CRT landmark experiment

---

## Key Results

| Result | Evidence |
|--------|----------|
| Modular arithmetic emerges from MDL | Figure 1: ratio drops 1.0→0.004 |
| Multi-agent consensus detects real rules | Noise: 0 axioms (no false positives) |
| Proof market prevents bad modifications | 98% rejection rate for random proposals |
| Convergence in ~8 rounds | Table 3 |
| CRT derived from joint compression | Figure 3 |

---

## Papers

1. **Mathematical Structure Emergence in MDL-Optimal Agent Societies**
   → `docs/papers/paper1_mathematical_emergence.md`
   Target: NeurIPS / ICLR

2. **Adversarial Self-Modification via Commit-Reveal Proof Markets**
   → `docs/papers/paper2_proof_market.md`
   Target: ICML / NeurIPS

---

## Project Structure
ouroboros/
├── core/          config, phase1_runner, phase2_runner, phase3_runner
├── environment/   6 observation environments + joint_environment
├── compression/   MDL engine, beam search, MCMC, hierarchical MDL
├── agents/        BaseAgent → SynthesisAgent → HierarchicalAgent
│                  → TheoryAgent → SelfModifyingAgent
├── proof_market/  commit_reveal, counterexample, market, ood_pressure
└── emergence/     proto_axiom_pool, scale_axiom_pool, causal_theory,
crt_detector

---

## References

- Rissanen, J. (1978). Modeling by shortest data description.
- Schmidhuber, J. (2003). Gödel machines: Fully self-referential optimal agents.
- Grünwald, P. (2007). The Minimum Description Length Principle.
- Hoel, E.P. et al. (2013). Quantifying causal emergence.
- Li, M. & Vitányi, P. (1997). An Introduction to Kolmogorov Complexity.