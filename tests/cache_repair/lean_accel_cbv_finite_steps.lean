/-
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Lean

/-!
Additive negative fixture: both tactics must report the actual maximum-step
error and Lean must exit nonzero. A different error is not a passing control.
Do not publish any failed-elaboration output as a proof artifact.
-/

set_option maxHeartbeats 0
set_option maxRecDepth 0
set_option cbv.maxSteps 10

example : (List.replicate 10 1).length = 10 := by
  cbv

example : (List.replicate 10 1).length = 10 := by
  decide_cbv
