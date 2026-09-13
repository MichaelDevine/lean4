/-
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Lean

/-!
Additive fixture for explicit zero-as-unlimited Sym.Simp/cbv steps.
Run with the actual rebuilt pinned Lean and no command-line option overrides.
This file must succeed; its explicit-zero proofs must fail on the unchanged
pin. The separate finite-limit and false-proposition files must fail.
-/

set_option maxHeartbeats 0
set_option maxRecDepth 0

namespace LeanAccel.CbvUnlimitedSteps

theorem cbvDefault : (List.replicate 10 1).length = 10 := by
  cbv

theorem decideCbvDefault : (List.replicate 10 1).length = 10 := by
  decide_cbv

set_option cbv.maxSteps 0 in
theorem cbvZero : (List.replicate 10 1).length = 10 := by
  cbv

set_option cbv.maxSteps 0 in
theorem decideCbvZero : (List.replicate 10 1).length = 10 := by
  decide_cbv

#print axioms cbvDefault
#print axioms decideCbvDefault
#print axioms cbvZero
#print axioms decideCbvZero

end LeanAccel.CbvUnlimitedSteps
