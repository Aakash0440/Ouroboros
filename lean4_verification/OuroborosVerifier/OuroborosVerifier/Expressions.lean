-- OUROBOROS Expression Library for Lean4
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

namespace OUROBOROS

inductive Expr : Type where
  | Const  : ℕ → Expr
  | Time   : Expr
  | Add    : Expr → Expr → Expr
  | Sub    : Expr → Expr → Expr
  | Mul    : Expr → Expr → Expr
  | Mod    : Expr → Expr → Expr
  | Div    : Expr → Expr → Expr
  deriving Repr

def eval (e : Expr) (t : ℕ) : ℕ :=
  match e with
  | Expr.Const n => n
  | Expr.Time    => t
  | Expr.Add l r => eval l t + eval r t
  | Expr.Sub l r => eval l t - eval r t
  | Expr.Mul l r => eval l t * eval r t
  | Expr.Mod l r => if eval r t = 0 then 0 else eval l t % eval r t
  | Expr.Div l r => if eval r t = 0 then 0 else eval l t / eval r t

def linearModular (slope intercept modulus : ℕ) : Expr :=
  Expr.Mod (Expr.Add (Expr.Mul (Expr.Const slope) Expr.Time)
                     (Expr.Const intercept))
           (Expr.Const modulus)

theorem linearModular_correct (s i m : ℕ) (hm : 0 < m) (t : ℕ) :
    eval (linearModular s i m) t = (s * t + i) % m := by
  simp [linearModular, eval]; omega

theorem linearModular_periodic (s i m : ℕ) (hm : 0 < m) (t : ℕ) :
    eval (linearModular s i m) (t + m) = eval (linearModular s i m) t := by
  simp only [linearModular_correct hm]
  rw [show s * (t + m) + i = (s * t + i) + s * m by ring]
  rw [Nat.add_mul_mod_self_right]

def matchesStream (expr : Expr) (stream : List ℕ) (alpha : ℕ) : Prop :=
  ∀ t : Fin stream.length,
    eval expr t.val % alpha = stream.get t

def soundAxiom (expr : Expr) (streams : List (List ℕ)) (alpha : ℕ) : Prop :=
  ∀ stream ∈ streams, matchesStream expr stream alpha

end OUROBOROS
