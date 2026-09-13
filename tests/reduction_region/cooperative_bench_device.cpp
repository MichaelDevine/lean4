/* Actual-checker expression benchmark, not a whole-proof performance claim.
   Arguments select fixture sizes only; they are not evaluator limits. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
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
static expr binary(char const * op, expr const & a, expr const & b) {
    return mk_app(mk_app(mk_const(name({"Nat", op})), a), b);
}

int main(int argc, char ** argv) {
    if (argc > 3) return 2;
    auto bits = argc > 1 ? std::stoull(argv[1]) : 16384;
    auto repeats = argc > 2 ? std::stoull(argv[2]) : 3;
    require(bits >= 4 && repeats > 0, "use at least four fixture bits and one repetition");
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
    nat base(2u), exponent(static_cast<unsigned long>(bits));
    nat value(nat_pow(base.raw(), exponent.raw()));
    expr a = mk_lit(literal(value - nat(1u))), b = mk_lit(literal(value - nat(3u)));
    expr c = mk_lit(literal(value - nat(5u))), d = mk_lit(literal(value - nat(7u)));
    expr input = binary("sub", binary("add", binary("mul", a, b), binary("mul", c, d)), a);
    type_checker reference(env);
    expr expected = reference.whnf(input);
    reduction_sycl_backend scalar(reduction_sycl_execution::scalar), cooperative;
    auto admission = [](reduction_session const & s, outcome reason) {
        // Data-derived storage for this four-operation fixture. No retry,
        // truncation, timeout or production scheduling policy is introduced.
        if (reason != outcome::running) return optional<reduction_workspace>();
        return optional<reduction_workspace>({s.program_size(), 0, s.program_size(),
            s.literal_size() * 2 + 4, s.literal_size()});
    };
    closure_reduction_extension scalar_extension(scalar, admission);
    closure_reduction_extension cooperative_extension(cooperative, admission);
    using clock = std::chrono::steady_clock;
    auto run = [&](unsigned mode, char const * phase, unsigned long long round) {
        auto start = clock::now();
        expr result;
        {
            type_checker checker(env); // fresh cache in every timed path
            scope_reduction_extension enable(mode == 0 ? nullptr :
                (mode == 1 ? static_cast<reduction_extension *>(&scalar_extension) : &cooperative_extension));
            result = checker.whnf(input);
        }
        auto end = clock::now();
        require(is_bi_equal(result, expected), "timed checker result differs from CPU reference");
        if (mode != 0) {
            auto & extension = mode == 1 ? scalar_extension : cooperative_extension;
            require(extension.last_attempt() == reduction_attempt::complete, "timed GPU reduction fell back");
        }
        if (mode == 2)
            require(cooperative.work_group_size() > 1 && cooperative.last_cooperative_products() == 2,
                    "timed cooperative products were not distributed");
        std::printf("phase=%s round=%llu mode=%s bits=%llu elapsed_ms=%.6f group_size=%zu products=%llu\n",
            phase, round, mode == 0 ? "cpu" : (mode == 1 ? "gpu-scalar" : "gpu-cooperative"), bits,
            std::chrono::duration<double, std::milli>(end - start).count(),
            mode == 0 ? 0 : (mode == 1 ? scalar.work_group_size() : cooperative.work_group_size()),
            static_cast<unsigned long long>(mode == 2 ? cooperative.last_cooperative_products() : 0));
    };
    for (unsigned mode = 0; mode < 3; ++mode) run(mode, "cold", 0);
    for (unsigned long long round = 0; round < repeats; ++round)
        for (unsigned offset = 0; offset < 3; ++offset) run((round + offset) % 3, "warm", round);
    std::puts("actual-checker cooperative benchmark checks passed");
}
