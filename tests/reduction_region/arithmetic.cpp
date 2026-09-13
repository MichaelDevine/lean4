/* Additive demanded-expression arithmetic and checker integration controls. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#include "library/closure_reduction_extension.h"
#ifdef LEAN_REDUCTION_ARITHMETIC_GPU
#include "library/reduction_sycl.h"
#endif
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

template<typename Backend> struct counting_backend {
    Backend & m_backend;
    unsigned m_calls = 0;
    outcome operator()(std::vector<instruction> const & code, machine & state,
        std::vector<closure> & args, std::vector<binding> & bindings, arithmetic_workspace & numeric) {
        ++m_calls;
        return m_backend(code, state, args, bindings, numeric);
    }
};

template<typename Backend>
static void run_arithmetic_controls(Backend & backend, environment const & env) {
    expr type = mk_const(name("Nat"));
    expr zero = mk_lit(literal(0u)), one = mk_lit(literal(1u)), seven = mk_lit(literal(7u));
    expr large = mk_lit(literal(nat("340282366920938463463374607431768211457")));
    expr ctor_zero = mk_const(name({"Nat", "zero"}));
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr first = mk_lambda(name("x"), type,
        mk_lambda(name("y"), type, mk_bvar(1), binder_info::Default), binder_info::Default);
    expr unused = mk_app(mk_mvar(name("unused")), mk_bvar(999));
    expr product = binary("mul", large, large);
    expr nested = binary("sub", binary("add", product, large), seven);
    expr body = binary("add", mk_bvar(0), binary("mul", mk_bvar(0), seven));
    expr captured = mk_let(name("x"), type, large,
        mk_let(name("f"), type, mk_lambda(name("y"), type,
            binary("mul", mk_bvar(1), mk_bvar(0)), binder_info::Default),
            mk_let(name("x"), type, seven, mk_app(mk_bvar(1), one), false), false), false);
    std::vector<expr> cases{
        binary("add", zero, zero), binary("add", large, large),
        binary("sub", seven, large), binary("sub", large, seven),
        product, nested, binary("mul", ctor_zero, large), binary("add", ctor_zero, seven),
        mk_app(identity, ctor_zero),
        mk_let(name("x"), type, product, body, false), captured,
        mk_app(mk_app(first, nested), unused),
        mk_let(name("unused"), type, unused, nested, false),
        mk_app(first, nested), mk_app(first, mk_bvar(0))
    };
    nat wide_base(2u), wide_exponent(16384u);
    nat wide_value(nat_pow(wide_base.raw(), wide_exponent.raw()));
    expr wide = mk_lit(literal(wide_value - nat(1u)));
    cases.push_back(binary("mul", wide, wide));
    cases.push_back(binary("sub", binary("add", binary("mul", wide, wide), wide), seven));
    expr chain = large;
    for (unsigned i = 0; i < 128; ++i)
        chain = binary("add", binary("mul", chain, seven), large);
    cases.push_back(chain);
    counting_backend counted{backend};
    auto admission = [](reduction_session const & session, outcome reason) {
        // Explicit fixture sizing, not a runtime default or ceiling. This
        // control requires the nested chain to complete in one submission.
        auto count = session.program_size();
        if (reason == outcome::running)
            return optional<reduction_workspace>({count, count, count,
                count * (session.literal_size() + 1) * 4, count * (session.literal_size() + 1) * 4});
        return optional<reduction_workspace>();
    };
    closure_reduction_extension extension(counted, admission);
    unsigned checks = 0;
    for (auto const & input : cases) {
        type_checker cpu(env), accelerated(env);
        expr expected = cpu.whnf(input);
        auto before = counted.m_calls;
        expr actual;
        {
            scope_reduction_extension enable(&extension);
            actual = accelerated.whnf(input);
        }
        require(extension.last_attempt() == reduction_attempt::complete && counted.m_calls == before + 1,
                "arithmetic region did not complete in one backend submission");
        require(is_bi_equal(actual, expected), "actual checker arithmetic result differs from CPU");
        ++checks;
    }
#ifdef LEAN_REDUCTION_ARITHMETIC_GPU
#ifdef LEAN_REDUCTION_ARITHMETIC_SCALAR
    require(backend.work_group_size() == 1 && backend.last_cooperative_products() == 0,
            "scalar GPU control used cooperative execution");
#else
    require(backend.work_group_size() > 1 && backend.last_cooperative_products() == 128,
            "nested arithmetic did not actually distribute its products");
#endif
    std::printf("gpu_group_size=%zu cooperative_products=%llu\n", backend.work_group_size(),
                static_cast<unsigned long long>(backend.last_cooperative_products()));
#endif

    // The same primitive must not be numerically reduced by core WHNF.
    for (unsigned flags = 0; flags < 4; ++flags) {
        type_checker cpu(env), accelerated(env);
        expr expected = cpu.whnf_core(nested, flags & 1, flags & 2);
        scope_reduction_extension enable(&extension);
        expr actual = accelerated.whnf_core(nested, flags & 1, flags & 2);
        require(is_bi_equal(actual, expected) && !is_nat_lit(actual), "core reduction acquired full arithmetic semantics");
        ++checks;
    }

    // Genuine shortage/recovery: preserve computed operands through repeated
    // frame/result allocation, including an injected failed submission.
    reduction_session session(nested, 0, 0, reduction_mode::full(), &env);
    bool saw_frames = false, saw_values = false, saw_columns = false, injected = false;
    while (true) {
        auto status = session.advance(backend);
        if (status == outcome::complete) break;
        require(status == outcome::need_arguments || status == outcome::need_bindings ||
                status == outcome::need_frames || status == outcome::need_values || status == outcome::need_columns,
                "unexpected arithmetic continuation outcome");
        saw_frames |= status == outcome::need_frames;
        saw_values |= status == outcome::need_values;
        saw_columns |= status == outcome::need_columns;
        if (status == outcome::need_values && session.state().m_num_values != 0 && !injected) {
            expr before = session.reconstruct();
            auto used = session.state().m_num_values;
            auto fail = [](auto const &, auto & state, auto &, auto &, auto & numeric) -> outcome {
                state.m_has_value = false;
                numeric.m_values.clear(); numeric.m_frames.clear();
                throw reduction_backend_error("injected arithmetic transfer failure");
            };
            bool caught = false;
            try { session.advance(fail); } catch (reduction_backend_error const &) { caught = true; }
            require(caught && session.state().m_num_values == used && is_bi_equal(session.reconstruct(), before),
                    "failed arithmetic submission changed its checkpoint");
            auto corrupt = [](auto const &, auto & state, auto &, auto &, auto &) {
                state.m_has_value = true; state.m_value = {UINT64_MAX, 1, true};
                return outcome::running;
            };
            require(session.advance(corrupt) == outcome::invalid && is_bi_equal(session.reconstruct(), before),
                    "invalid arithmetic result reference committed");
            bool rejected = false;
            try { session.resize_workspace(session.argument_capacity(), session.binding_capacity(),
                                            session.frame_capacity(), used - 1); }
            catch (std::invalid_argument const &) { rejected = true; }
            require(rejected, "live arithmetic data was discarded on resize");
            injected = true;
        }
        session.resize_workspace(
            session.argument_capacity() + (status == outcome::need_arguments),
            session.binding_capacity() + (status == outcome::need_bindings),
            session.frame_capacity() + (status == outcome::need_frames),
            std::max<reduction::index>(session.value_capacity(), session.state().m_required_values),
            std::max<reduction::index>(session.column_capacity(), session.state().m_required_columns));
    }
    require(saw_frames && saw_values && injected, "arithmetic recovery paths were not exercised");
#if defined(LEAN_REDUCTION_ARITHMETIC_GPU) && !defined(LEAN_REDUCTION_ARITHMETIC_SCALAR)
    require(saw_columns, "cooperative scratch recovery was not exercised");
#else
    require(!saw_columns, "scalar arithmetic unexpectedly demanded parallel scratch");
#endif
    type_checker cpu(env);
    require(is_bi_equal(session.reconstruct(), cpu.whnf(nested)), "resumed arithmetic changed the result");

    // A nonnumeric first operand declines before demanding a divergent RHS.
    // This is intentionally malformed syntax, not a well-typed proof claim.
    expr self = mk_lambda(name("x"), type, mk_app(mk_bvar(0), mk_bvar(0)), binder_info::Default);
    reduction_session lazy(binary("add", mk_sort(mk_level_zero()), mk_app(self, self)), 2, 0,
                           reduction_mode::full(), &env);
    lazy.resize_workspace(2, 0, 1, 0);
    require(lazy.advance(backend) == outcome::unsupported, "nonnumeric left operand demanded its divergent RHS");

    for (expr input : {binary("div", large, seven), mk_app(mk_const(name({"Nat", "add"})), one),
                       mk_app(binary("add", one, one), one)}) {
        require(!extension.reduce(env, local_ctx(), input, reduction_mode::full()) &&
                extension.last_attempt() == reduction_attempt::unsupported,
                "unsupported primitive or arity became a successful reduction");
    }
    // A checked declaration in an independent environment can have this
    // name without being the builtin constructor. Its full normal form must
    // follow that environment, not syntax-only primitive recognition.
    expr sort0 = mk_sort(mk_level_zero()), sort1 = mk_sort(mk_level_one());
    auto shadow = env.add(mk_definition(name({"Nat", "zero"}), names(), sort1, sort0,
                                      reducibility_hints::mk_abbreviation()));
    type_checker shadow_cpu(shadow);
    require(shadow_cpu.whnf(ctor_zero) == sort0, "checked shadow environment setup failed");
    require(!extension.reduce(shadow, local_ctx(), ctor_zero, reduction_mode::full()),
            "constructor recognition ignored an environment definition");
    expr typed_identity = mk_lambda(name("x"), sort1, mk_bvar(0), binder_info::Default);
    type_checker shadow_accelerated(shadow);
    {
        scope_reduction_extension enable(&extension);
        require(shadow_accelerated.whnf(mk_app(typed_identity, ctor_zero)) == sort0 &&
                shadow_accelerated.check(mk_app(typed_identity, ctor_zero)) == sort1,
                "environment-bound CPU fallback changed a well-typed expression");
    }
    std::printf("actual-expression arithmetic checks passed: %u; nested_operations=256; single_submission=yes\n", checks);
}

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
#ifdef LEAN_REDUCTION_ARITHMETIC_GPU
#ifdef LEAN_REDUCTION_ARITHMETIC_SCALAR
    reduction_sycl_backend backend(reduction_sycl_execution::scalar);
#else
    reduction_sycl_backend backend;
#endif
#else
    reduction_cpu_backend backend;
#endif
    run_arithmetic_controls(backend, env);
}
