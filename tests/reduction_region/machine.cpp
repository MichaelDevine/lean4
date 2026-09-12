/* Additive tests for actual-expression compilation and lazy continuations. */
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "runtime/init_module.h"
#include "kernel/reduction_program.h"

extern "C" void lean_initialize();
using namespace lean;
using namespace lean::reduction;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static expr execute(expr const & input, bool grow) {
    reduction_region region(input);
    auto program = make_reduction_program(region);
    machine m(region.root());
    std::vector<closure> args(grow ? 0 : 64);
    std::vector<binding> bindings(grow ? 0 : 64);
    for (;;) {
        machine before = m;
        auto result = m.step(program.data(), program.size(), args.data(), args.size(),
                             bindings.data(), bindings.size());
        if (result == outcome::running) continue;
        if (result == outcome::need_arguments || result == outcome::need_bindings) {
            require(m.m_control.m_code == before.m_control.m_code &&
                    m.m_control.m_env == before.m_control.m_env &&
                    m.m_num_arguments == before.m_num_arguments &&
                    m.m_num_bindings == before.m_num_bindings,
                    "resource boundary changed the suspended machine");
            if (result == outcome::need_arguments) args.resize(args.size() * 2 + 1);
            else bindings.resize(bindings.size() * 2 + 1);
            continue;
        }
        require(result == outcome::complete, "supported computation did not complete");
        auto const & n = program[m.m_control.m_code];
        require(n.m_op == opcode::value, "expected a closed literal result");
        return region.get(n.m_source).m_expr;
    }
}

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    static_assert(std::is_trivially_copyable<instruction>::value, "instruction is not transferable");
    static_assert(std::is_trivially_copyable<closure>::value, "closure is not transferable");
    static_assert(std::is_trivially_copyable<binding>::value, "binding is not transferable");
    static_assert(std::is_trivially_copyable<machine>::value, "machine is not transferable");
    expr type = mk_const(name("Nat"));
    expr one = mk_lit(literal(11u)), two = mk_lit(literal(22u));
    expr unused = mk_app(mk_mvar(name("unsupported")), mk_bvar(999));
    expr first = mk_lambda(name("x"), type,
        mk_lambda(name("y"), type, mk_bvar(1), binder_info::Default), binder_info::Default);
    expr choose = mk_app(mk_app(first, one), unused);
    expr unused_let = mk_let(name("x"), type, unused, one, false);
    expr captured = mk_let(name("x"), type, one,
        mk_let(name("f"), type, mk_lambda(name("y"), type, mk_bvar(1), binder_info::Default),
            mk_let(name("x"), type, two, mk_app(mk_bvar(1), two), false), false), false);
    for (bool grow : {false, true}) {
        require(execute(choose, grow) == one, "unused argument was demanded");
        require(execute(unused_let, grow) == one, "unused let was demanded");
        require(execute(captured, grow) == one, "lexical binding was replaced by caller binding");
        expr large = mk_lit(literal(nat("340282366920938463463374607431768211457")));
        require(execute(mk_app(mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default), large), grow)
                    == large, "opaque exact literal payload changed");
    }

    reduction_region boundary(unused);
    auto p = make_reduction_program(boundary);
    machine m(boundary.root()); closure args[1]; binding bindings[1];
    require(m.step(p.data(), p.size(), args, 1, bindings, 1) == outcome::running, "app did not advance");
    require(m.step(p.data(), p.size(), args, 1, bindings, 1) == outcome::unsupported,
            "unsupported head was treated as a result");

    instruction bad[]{{opcode::bound, 0, 0, 0}};
    machine corrupted(0); corrupted.m_num_bindings = 1; corrupted.m_control.m_env = 0;
    bindings[0] = {{0, no_binding}, 0};
    require(corrupted.step(bad, 1, args, 1, bindings, 1) == outcome::invalid,
            "cyclic environment was accepted");
    instruction invalid[]{{opcode::app, 99, 0, 0}};
    machine invalid_machine(0);
    require(invalid_machine.step(invalid, 1, args, 1, bindings, 1) == outcome::invalid,
            "invalid program edge was accepted");
    std::puts("lazy reduction machine checks passed");
}
