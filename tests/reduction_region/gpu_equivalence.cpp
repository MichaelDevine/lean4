/* Additive actual-checker CPU/GPU closure equivalence controls.
   Timings include capture, transfers, dispatch, continuation and reconstruction.
   These small controls are not representative whole-proof benchmarks. */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#include "library/closure_reduction_extension.h"
#include "library/reduction_sycl.h"
extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);
using namespace lean;
using namespace lean::reduction;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
    expr type = mk_sort(mk_level_one());
    expr one = mk_lit(literal(11u)), two = mk_lit(literal(22u));
    expr unused = mk_app(mk_mvar(name("unused")), mk_bvar(999));
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr first = mk_lambda(name("x"), type,
        mk_lambda(name("y"), type, mk_bvar(1), binder_info::Implicit), binder_info::Default);
    expr large = mk_lit(literal(nat("340282366920938463463374607431768211457")));
    expr captured = mk_let(name("x"), type, one,
        mk_let(name("f"), type, mk_lambda(name("y"), type, mk_bvar(1), binder_info::Default),
            mk_let(name("x"), type, two, mk_app(mk_bvar(1), two), false), false), false);
    expr deep = one;
    for (unsigned i = 0; i < 128; ++i) deep = mk_app(identity, deep);
    std::vector<expr> cases{
        mk_app(identity, one),
        mk_app(mk_app(first, one), unused),
        mk_let(name("unused"), type, unused, one, false),
        captured, mk_app(first, one), mk_app(first, mk_bvar(0)),
        mk_app(identity, large), deep
    };
    reduction_sycl_backend gpu;
    auto admission = [](reduction_session const & s, outcome reason) {
        // Explicit control policy: start empty and grow only on a workspace
        // outcome. Geometric growth exercises continuation without a ceiling.
        auto a = s.argument_capacity(), b = s.binding_capacity();
        if (reason == outcome::need_arguments) a = a * 2 + 1;
        if (reason == outcome::need_bindings) b = b * 2 + 1;
        return optional<reduction_workspace>(reduction_workspace{a, b});
    };
    closure_reduction_extension extension(gpu, admission);
    using clock = std::chrono::steady_clock;
    unsigned checks = 0;
    for (unsigned mode = 0; mode < 5; ++mode) {
        for (std::size_t i = 0; i < cases.size(); ++i) {
            auto reduce = [&](type_checker & tc) {
                if (mode == 0) return tc.whnf(cases[i]);
                return tc.whnf_core(cases[i], (mode - 1) & 1, (mode - 1) & 2);
            };
            type_checker cpu(env), accelerated(env);
            auto cpu_start = clock::now();
            expr expected = reduce(cpu);
            auto cpu_end = clock::now();
            auto gpu_start = clock::now();
            expr actual;
            {
                scope_reduction_extension enable(&extension);
                actual = reduce(accelerated);
            }
            auto gpu_end = clock::now();
            require(extension.last_attempt() == reduction_attempt::complete,
                    "a GPU case fell back or failed");
            require(is_bi_equal(actual, expected), "CPU/GPU expression or binder annotations differ");
            ++checks;
            std::printf("case=%zu mode=%u cpu_ns=%lld gpu_ns=%lld\n", i, mode,
                static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count()));
        }
    }
    // This is deliberately an expression-reduction matrix, not a claim that
    // the malformed unused/open controls are well-typed proof terms.
    std::printf("actual-checker GPU equivalence checks passed: %u\n", checks);
}
