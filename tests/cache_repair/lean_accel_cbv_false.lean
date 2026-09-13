/-
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Lean

/-!
Additive negative fixture under explicit unlimited steps. cbv must leave an
unsolved false goal, decide_cbv must report false evaluation, and Lean must
exit nonzero. A step-limit, syntax or loader error is not a passing control.
Do not publish any failed-elaboration output as a proof artifact.
-/

set_option maxHeartbeats 0
set_option maxRecDepth 0
set_option cbv.maxSteps 0

example : (2 : Nat) + 2 = 5 := by
  cbv

example : (2 : Nat) + 2 = 5 := by
  decide_cbv
