# feat(NumberTheory): Linear modular surjectivity and CRT instances

## Mathematical Content

This PR adds two files to `Mathlib/NumberTheory/`:

### `LinearModularSurjective.lean`

Lemmas about linear functions `t ↦ (slope * t + intercept) % N`:

1. **`LinearMod.periodic`** — periodicity with period N (proved by `omega`)
2. **`LinearMod.range_bound`** — output is always < N
3. **`LinearMod.ax00001_satisfies_spec`** — the specific instance (3t+1) % 7 is periodic, bounded, and surjective onto {0,...,6}, with explicit witnesses for each residue
4. **`LinearMod.surjective_of_coprime`** — when slope and N are coprime, the function is surjective (general case)

### `CRTInstances.lean`

Concrete instances of the Chinese Remainder Theorem for (mod 7, mod 11):

1. **`CRT711.coprime`** — gcd(7,11) = 1
2. **`CRT711.witness_mod7/11`** — verification that `(a*22 + b*56) % 77` gives the correct residues
3. **`CRT711.existence`** — existence of CRT solution for all (a < 7, b < 11) pairs
4. **`CRT711.uniqueness`** — uniqueness of solution mod 77

## Motivation

These lemmas were **machine-discovered** by the OUROBOROS multi-agent system,
which uses MDL (Minimum Description Length) compression pressure to discover
mathematical laws from integer sequences. The system found `(3*t+1) % 7`
as its first promoted axiom when 8 independent agents all converged on
the same expression for the sequence 1, 4, 0, 3, 6, 2, 5, 1, 4, 0...

The Lean4 proofs verify that the discovered expressions have the
mathematical properties the agents claimed to find.

## Proof Strategy

- `omega` handles all linear arithmetic over ℕ and ℤ
- `norm_num` verifies concrete arithmetic (e.g., 56 % 11 = 1)
- `interval_cases` splits on the 7 possible residue values for surjectivity
- Explicit witnesses avoid any choice-based reasoning

## Style Notes

- No `sorry` placeholders — all proofs are complete
- Names follow Mathlib4 `camelCase` conventions
- Every theorem has a docstring
- Imports are minimal (no unused imports)
- Proofs are readable (not golfed)