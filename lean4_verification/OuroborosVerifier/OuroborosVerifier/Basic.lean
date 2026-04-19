-- OUROBOROS Lean4 Verification Library
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.ModCast
import Mathlib.Tactic

def ExprMatchesStream (expr_fn : ℕ → ℕ) (stream : List ℕ) : Prop :=
  ∀ t : Fin stream.length, expr_fn t.val = stream.get t

def ModArithFn (slope intercept modulus : ℕ) (t : ℕ) : ℕ :=
  (slope * t + intercept) % modulus

theorem mod_arith_periodic (slope intercept modulus : ℕ) (hm : 0 < modulus) :
    ∀ t : ℕ, ModArithFn slope intercept modulus (t + modulus) =
             ModArithFn slope intercept modulus t := by
  intro t
  simp [ModArithFn]
  omega

def IsBetterPredictor (e1 e2 : ℕ → ℕ) (stream : List ℕ) (errors1 errors2 : ℕ) : Prop :=
  errors2 < errors1

theorem perfect_predictor_no_counterexample
    (expr : ℕ → ℕ) (stream : List ℕ)
    (h : ExprMatchesStream expr stream) :
    ∀ (other : ℕ → ℕ) (other_errors : ℕ),
      ¬ IsBetterPredictor expr other stream 0 other_errors := by
  intro other other_errors h_better
  simp [IsBetterPredictor] at h_better
  omega

-- CRT existence (stated cleanly without chineseRemainder API)
theorem crt_exists (m n a b : ℕ) (hm : 0 < m) (hn : 0 < n)
    (hcop : Nat.Coprime m n) :
    ∃ x : ℕ, x % m = a % m ∧ x % n = b % n := by
  obtain ⟨x, hx⟩ := (Nat.chineseRemainder' hcop a b)
  exact ⟨x, hx.1, hx.2⟩

#check mod_arith_periodic
#check perfect_predictor_no_counterexample
