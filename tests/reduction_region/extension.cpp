/* Additive controls linked with the modified type_checker implementation. */
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include "runtime/init_module.h"
#include "runtime/thread.h"
#include "kernel/type_checker.h"
#include "kernel/reduction_extension.h"
#include "kernel/reduction_session.h"

extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);
using namespace lean;
using namespace lean::reduction;
static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct closure_extension : reduction_extension {
    unsigned calls = 0;
    bool core = false, cheap_rec = false, cheap_proj = false;
    optional<expr> reduce(environment const &, local_ctx const &, expr const & e,
                          reduction_mode mode) override {
        ++calls; core = mode.is_core(); cheap_rec = mode.cheap_rec(); cheap_proj = mode.cheap_proj();
        reduction_session session(e, 0, 0);
        reduction_cpu_backend cpu;
        while (true) {
            switch (session.advance(cpu)) {
            case outcome::complete: return some_expr(session.reconstruct());
            case outcome::need_arguments:
                session.resize_workspace(session.argument_capacity() + 1, session.binding_capacity()); break;
            case outcome::need_bindings:
                session.resize_workspace(session.argument_capacity(), session.binding_capacity() + 1); break;
            default: return none_expr();
            }
        }
    }
};

struct consulting_extension : reduction_extension {
    unsigned calls = 0;
    optional<expr> reduce(environment const & env, local_ctx const & lctx, expr const & e,
                          reduction_mode mode) override {
        ++calls;
        type_checker cpu(env, lctx);
        return some_expr(mode.is_core() ? cpu.whnf_core(e, mode.cheap_rec(), mode.cheap_proj()) : cpu.whnf(e));
    }
};

struct declining_extension : reduction_extension {
    unsigned calls = 0;
    optional<expr> reduce(environment const &, local_ctx const &, expr const &, reduction_mode) override {
        ++calls;
        return none_expr();
    }
};

struct throwing_extension : reduction_extension {
    unsigned calls = 0;
    optional<expr> reduce(environment const &, local_ctx const &, expr const &, reduction_mode) override {
        ++calls;
        throw std::runtime_error("injected extension exception");
    }
};

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    lean_object * created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "could not create the test environment");
    lean_object * elab_env = lean_io_result_get_value(created);
    lean_inc(elab_env);
    lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(elab_env));
    expr type = mk_sort(mk_level_one());
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr value = mk_sort(mk_level_zero());
    expr input = mk_app(identity, value);
    closure_extension extension;
    {
        scope_reduction_extension enable(&extension);
        type_checker tc(env);
        require(tc.whnf(input) == value && extension.calls == 1 && !extension.core,
                "full reduction did not dispatch the original expression");
        require(tc.whnf(input) == value && extension.calls == 1, "cache hit redispatched");
        require(tc.check(input) == type, "well-typed application failed checking");
        require(!tc.is_def_eq(type, value), "distinct universes became equal with extension");
        for (unsigned flags = 0; flags < 4; ++flags) {
            type_checker core(env);
            require(core.whnf_core(input, flags & 1, flags & 2) == value,
                    "core reduction changed result");
            require(extension.core && extension.cheap_rec == bool(flags & 1) &&
                    extension.cheap_proj == bool(flags & 2), "core mode flags changed");
        }
        unsigned before = extension.calls;
        local_ctx empty_locals;
        bool thread_declined = false;
        std::thread other([&]() {
            lean_initialize_thread();
            thread_declined = !try_reduction_extension(env, empty_locals, input, reduction_mode::full());
            lean_finalize_thread();
        });
        other.join();
        require(thread_declined && extension.calls == before, "extension leaked to another thread");
        {
            scope_reduction_extension disable(nullptr);
            type_checker cpu(env);
            require(cpu.whnf(input) == value && extension.calls == before, "nested disable failed");
        }
        type_checker restored(env);
        require(restored.whnf(input) == value && extension.calls == before + 1, "scope did not restore");
    }
    unsigned before = extension.calls;
    type_checker disabled(env);
    require(disabled.whnf(input) == value && extension.calls == before, "extension escaped scope");
    consulting_extension consulting;
    {
        scope_reduction_extension enable(&consulting);
        type_checker tc(env);
        require(tc.whnf(input) == value && consulting.calls == 1, "CPU consultation redispatched");
    }
    require(!disabled.is_def_eq(value, type), "distinct universes became equal");
    declining_extension decline;
    {
        scope_reduction_extension enable(&decline);
        type_checker tc(env);
        require(tc.whnf(input) == value && decline.calls > 0, "decline did not use CPU fallback");
    }
    throwing_extension throwing;
    {
        scope_reduction_extension enable(&throwing);
        type_checker tc(env);
        for (unsigned i = 0; i < 2; ++i) {
            bool caught = false;
            try { tc.whnf(input); } catch (std::runtime_error const &) { caught = true; }
            require(caught && throwing.calls == i + 1, "exception escaped scope restoration");
        }
    }
    std::puts("reduction extension checks passed");
}
