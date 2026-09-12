/* Additive host submission/recovery controls using actual Lean expressions. */
#include <cstdio>
#include <cstdlib>
#include "runtime/init_module.h"
#include "kernel/reduction_session.h"

extern "C" void lean_initialize();
using namespace lean;
using namespace lean::reduction;
static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    expr type = mk_const(name("Nat"));
    expr literal = mk_lit(lean::literal(19u));
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr input = mk_app(identity, literal);
    reduction_session session(input, 0, 0);
    reduction_cpu_backend cpu;
    require(session.advance(cpu) == outcome::need_arguments, "missing argument demand");
    require(is_bi_equal(session.reconstruct(), input), "resource stop changed input");
    session.resize_workspace(1, 0);
    require(session.advance(cpu) == outcome::need_bindings, "missing binding demand");
    require(is_bi_equal(session.reconstruct(), input), "suspended application changed");
    auto failing = [](auto const &, auto & state, auto &, auto &) -> outcome {
        state.m_control.m_code = no_binding;
        throw std::runtime_error("injected submission failure");
    };
    bool failed = false;
    try { session.advance(failing); } catch (std::runtime_error const &) { failed = true; }
    require(failed && is_bi_equal(session.reconstruct(), input), "failure corrupted checkpoint");
    auto invalid = [](auto const &, auto & state, auto &, auto &) {
        state.m_control.m_code = no_binding;
        return outcome::complete;
    };
    require(session.advance(invalid) == outcome::invalid &&
            is_bi_equal(session.reconstruct(), input), "invalid state committed");
    bool rejected = false;
    try { session.resize_workspace(0, 0); }
    catch (std::invalid_argument const &) { rejected = true; }
    require(rejected && session.argument_capacity() == 1, "live arguments discarded");
    session.resize_workspace(1, 1);
    require(session.advance(cpu) == outcome::complete && session.reconstruct() == literal,
            "resource continuation failed");

    expr unsupported = mk_app(mk_const(name("f")), literal);
    reduction_session boundary(unsupported, 1, 0);
    require(boundary.advance(cpu) == outcome::unsupported &&
            is_bi_equal(boundary.reconstruct(), unsupported), "unsupported became proof result");
    std::puts("reduction session checks passed");
}
