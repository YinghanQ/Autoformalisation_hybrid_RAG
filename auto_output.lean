import Mathlib.LinearAlgebra.Matrix.IsDiag
import Mathlib.LinearAlgebra.Matrix.Orthogonal

open Matrix

theorem orthogonal_columns_transpose_mul_self_is_diag
  {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n] [Field ℝ]
  (A : Matrix m n ℝ)
  (h : ∀ i j, i ≠ j → (transpose A * A) i j = 0) :
  IsDiag (transpose A * A) := by
  intro i j hij
  exact h i j hij
