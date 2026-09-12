/* Additive actual-checker adapter and fallback controls. */
#include <cstdio>
#include <cstdlib>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#include "library/closure_reduction_extension.h"

extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);
using namespace lean;
using namespace lean::reduction;
static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct backend_control {
    unsigned calls = 0;
    unsigned fail = 0;
    outcome operator()(std::vector<instruction> const & code, machine & state,
                        std::vector<closure> & arguments, std::vector<binding> & bindings) {
        ++calls;
        if (fail == 1) throw reduction_backend_error("injected device failure");
        if (fail == 2) throw std::bad_alloc();
        if (fail == 3) { state.m_control.m_code = no_binding; return outcome::complete; }
        if (fail == 4) throw std::logic_error("unexpected backend defect");
        return reduction_cpu_backend()(code, state, arguments, bindings);
    }
};

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment construction failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
    expr type = mk_sort(mk_level_one());
    expr value = mk_sort(mk_level_zero());
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr input = mk_app(identity, value);
    unsigned admissions = 0;
    auto admission = [&](reduction_session const & session, outcome reason) -> optional<reduction_workspace> {
        ++admissions;
        // One-slot increments intentionally exercise resource boundaries in
        // this small control; this is not the production admission policy.
        return optional<reduction_workspace>(reduction_workspace{
            session.argument_capacity() + (reason == outcome::need_arguments),
            session.binding_capacity() + (reason == outcome::need_bindings)});
    };
    backend_control backend;
    closure_reduction_extension extension(backend, admission);
    {
        scope_reduction_extension enable(&extension);
        type_checker tc(env);
        require(tc.whnf(input) == value && extension.last_attempt() == reduction_attempt::complete,
                "adapter did not complete through the checker");
        require(admissions >= 3 && backend.calls >= 3, "resource continuation not exercised");
        require(tc.check(input) == type && !tc.is_def_eq(value, type), "checking semantics changed");
        for (unsigned fail = 1; fail <= 3; ++fail) {
            backend.fail = fail;
            type_checker fresh(env);
            require(fresh.whnf(input) == value, "failure prevented exact CPU fallback");
            auto expected = fail == 1 ? reduction_attempt::device_failure :
                            fail == 2 ? reduction_attempt::memory_failure : reduction_attempt::invalid;
            require(extension.last_attempt() == expected, "failure reason was lost");
            if (fail == 1) require(extension.device_error() == "injected device failure", "device message lost");
        }
        backend.fail = 4;
        bool caught = false;
        try { type_checker fresh(env); fresh.whnf(input); }
        catch (std::logic_error const &) { caught = true; }
        require(caught, "unexpected implementation exception was hidden");
    }
    backend.fail = 0;
    auto decline = [](reduction_session const &, outcome) { return optional<reduction_workspace>(); };
    closure_reduction_extension declined(backend, decline);
    auto before = backend.calls;
    require(!declined.reduce(env, local_ctx(), input, reduction_mode::full()) &&
            declined.last_attempt() == reduction_attempt::declined && backend.calls == before,
            "declined work reached the backend");
    auto no_growth = [](reduction_session const &, outcome) {
        return optional<reduction_workspace>(reduction_workspace{0, 0});
    };
    closure_reduction_extension stalled(backend, no_growth);
    require(!stalled.reduce(env, local_ctx(), input, reduction_mode::full()) &&
            stalled.last_attempt() == reduction_attempt::invalid_admission,
            "non-growing admission did not terminate as an explicit resource-policy error");
    expr unsupported = mk_const(name("unsupported"));
    require(!extension.reduce(env, local_ctx(), unsupported, reduction_mode::full()) &&
            extension.last_attempt() == reduction_attempt::unsupported,
            "unsupported computation reported completion");
    std::puts("reduction adapter checks passed");
}
