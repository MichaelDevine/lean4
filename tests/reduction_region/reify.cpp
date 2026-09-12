/* Additive reconstruction controls using actual kernel substitution. */
#include <cstdio>
#include <cstdlib>
#include "runtime/init_module.h"
#include "kernel/reduction_reify.h"

extern "C" void lean_initialize();
using namespace lean;
using namespace lean::reduction;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static expr reduce(expr const & e, outcome expected) {
    reduction_region region(e);
    auto code = make_reduction_program(region);
    machine m(region.root()); closure args[32]; binding bindings[32];
    outcome status;
    do { status = m.step(code.data(), code.size(), args, 32, bindings, 32); }
    while (status == outcome::running);
    require(status == expected, "unexpected machine outcome");
    return reify_reduction_machine(region, m, args, 32, bindings, 32);
}

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    expr type = mk_const(name("Nat"));
    expr literal = mk_lit(lean::literal(11u));
    expr keep = mk_lambda(name("x"), type,
        mk_lambda(name("y"), type, mk_bvar(1), binder_info::Implicit), binder_info::Default);
    expr closed = mk_app(keep, literal);
    expr expected = instantiate(binding_body(keep), literal);
    require(is_bi_equal(reduce(closed, outcome::complete), expected), "closed lambda changed");

    expr open = mk_app(keep, mk_bvar(0));
    require(is_bi_equal(reduce(open, outcome::complete), instantiate(binding_body(keep), mk_bvar(0))),
            "open argument was captured by the result binder");

    expr body = mk_app(mk_app(mk_const(name("f")), mk_bvar(0)), mk_lit(lean::literal(22u)));
    expr suspended = mk_let(name("x"), type, literal, body, false);
    require(is_bi_equal(reduce(suspended, outcome::unsupported), instantiate(body, literal)),
            "suspended argument order or lexical substitution changed");

    reduction_region region(mk_bvar(0));
    auto lit = region.capture(literal);
    auto shared = region.capture(mk_app(mk_bvar(0), mk_bvar(0)));
    binding bindings[2]{{{lit, no_binding}, no_binding}, {{region.root(), 0}, 0}};
    reduction_reifier quote(region, bindings, 2);
    require(quote({region.root(), 1}) == literal, "nested closure did not reconstruct");
    expr duplicated = quote({shared, 1});
    require(is_eqp(app_fn(duplicated), app_arg(duplicated)), "shared quoted value was duplicated");

    bindings[0].m_parent = 0;
    bool rejected = false;
    try { reduction_reifier bad(region, bindings, 2); bad({region.root(), 0}); }
    catch (std::invalid_argument const &) { rejected = true; }
    require(rejected, "cyclic environment was accepted");
    std::vector<binding> deep(20000);
    deep[0] = {{lit, no_binding}, no_binding};
    for (std::size_t i = 1; i < deep.size(); ++i)
        deep[i] = {{region.root(), i - 1}, i - 1};
    reduction_reifier deep_quote(region, deep.data(), deep.size());
    require(deep_quote({region.root(), deep.size() - 1}) == literal,
            "deep environment reconstruction failed");
    std::puts("reduction reconstruction checks passed");
}
