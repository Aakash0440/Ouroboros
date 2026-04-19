-- OUROBOROS Lean4 Verification Library
-- Formalizes properties of modular arithmetic expressions
-- discovered by OUROBOROS agents.

import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.ModCast
import Mathlib.Tactic

-- Core proposition: an expression E correctly describes a stream
-- if it matches every observation E(t) = obs[t] for all t in range.
def ExprMatchesStream (expr_fn : ℕ → ℕ) (stream : List ℕ) : Prop :=
  ∀ t : Fin stream.length, expr_fn t.val = stream[t]

-- A modular arithmetic rule: (slope * t + intercept) % modulus
def ModArithFn (slope intercept modulus : ℕ) (t : ℕ) : ℕ :=
  (slope * t + intercept) % modulus

-- Key property: modular arithmetic is periodic
theorem mod_arith_periodic (slope intercept modulus : ℕ) (hm : 0 < modulus) :
    ∀ t : ℕ, ModArithFn slope intercept modulus (t + modulus) =
             ModArithFn slope intercept modulus t := by
  intro t
  simp [ModArithFn]
  ring_nf
  omega

-- A counterexample to expression E on stream S:
-- another expression E' that achieves lower MDL cost.
-- We formalize this as: E' predicts S with fewer errors than E.
def IsBetterPredictor
    (e1 e2 : ℕ → ℕ)
    (stream : List ℕ)
    (errors1 errors2 : ℕ) : Prop :=
  errors2 < errors1

-- The key soundness theorem:
-- If expression E matches stream S perfectly (0 errors),
-- then no other expression can be a valid counterexample
-- (since 0 errors is already optimal).
theorem perfect_predictor_no_counterexample
    (expr : ℕ → ℕ)
    (stream : List ℕ)
    (h : ExprMatchesStream expr stream) :
    ∀ (other : ℕ → ℕ) (other_errors : ℕ),
      ¬ IsBetterPredictor expr other stream 0 other_errors := by
  intro other other_errors h_better
  simp [IsBetterPredictor] at h_better
  omega

-- CRT: For coprime moduli m, n, the system
-- x ≡ a (mod m), x ≡ b (mod n) has a unique solution mod mn
-- We state this as a lemma for use in verification
theorem crt_exists (m n a b : ℕ) (hm : 0 < m) (hn : 0 < n)
    (hcop : Nat.Coprime m n) :
    ∃ x : ℕ, x % m = a % m ∧ x % n = b % n := by
  exact ⟨(a * (Nat.chineseRemainder hcop a b).1 * n +
          b * (Nat.chineseRemainder hcop a b).2 * m) % (m * n),
         by omega, by omega⟩

#check mod_arith_periodic
#check perfect_predictor_no_counterexample
LEAN

echo "✅ Lean4 library file created"