# Cache and evaluator repair controls

These additive controls exercise the maintained source directly. They are
separate from the unchanged original upstream test suite; passing these does
not substitute for building the complete toolchain and running that suite.

- `expr-cache-check.cpp`: actual expression equality, congruent fingerprints,
  structural collisions, all expression kinds and retained ownership.
- `mpz-content-check.cpp`: actual GMP and fallback integer content hashing.
- `expr-whnf-cache-check.cpp`: flat-cache growth, collisions, aliases, copies,
  moves and actual reference-count ownership.
- `expr-whnf-cache-allocation-check.cpp`: actual mimalloc allocation failures
  and retries; requires the link option `-Wl,--wrap=mi_new_n` and mimalloc.

Compile C++ controls using the selected full build's C++ flags and generated
headers, this source tree's headers, and its `libleancpp.a`, `libleanshared.so`
and configured GMP. Verify runtime linkage to the selected build. The fallback
integer control instead compiles the actual fallback integer sources with
`LEAN_USE_GMP` undefined and hides those symbols from the shared GMP runtime;
never pass its fallback integer objects into the GMP runtime.

Run the three Lean controls using the complete rebuilt toolchain and no
command-line option overrides. `lean_accel_cbv_unlimited_steps.lean` must
succeed and print the positive proofs' axioms. `lean_accel_cbv_finite_steps.lean`
must fail at both finite step limits. `lean_accel_cbv_false.lean` must reject
both false propositions for mathematical reasons, not a resource or loader
failure. The original upstream build must reject the explicit-zero positive
cases. No failed-elaboration artifact is a proof.

No original upstream test, expectation, registration or harness is changed.
