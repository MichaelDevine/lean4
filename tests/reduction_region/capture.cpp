/* Additive actual-Expr tests. No existing test or checker is replaced. */
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include "runtime/init_module.h"
#include "kernel/reduction_region.h"

extern "C" void lean_initialize();
using namespace lean;

static unsigned checks = 0;
static void require(bool condition, char const * message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    initialize_runtime_module();
    lean_initialize();
    lean_io_mark_end_initialization();

    expr type = mk_const(name("Nat"));
    expr value = mk_lit(literal(nat("340282366920938463463374607431768211457")));
    expr shared = mk_app(mk_const(name("f")), value);
    reduction_region graph(mk_app(shared, shared));
    require(graph.size() == 1, "capture traversed children eagerly");
    graph.expand(graph.root());
    auto root = graph.get(graph.root());
    require(root.m_num_children == 2, "application arity changed");
    require(root.m_children[0] == root.m_children[1], "sharing was lost");
    require(graph.size() == 2, "shared expression was duplicated");
    graph.expand(root.m_children[0]);
    auto child = graph.get(root.m_children[0]);
    require(is_eqp(graph.get(child.m_children[1]).m_expr, value), "large literal changed");
    auto size = graph.size();
    graph.expand(graph.root());
    require(graph.size() == size, "repeated expansion changed the graph");

    // This syntax is deliberately not executable. Capture must not attempt
    // type inference, normalization or evaluation of an unused let value.
    expr unused = mk_app(mk_mvar(name("notAssigned")), mk_bvar(nat(42)));
    reduction_region lazy(mk_let(name("x"), type, unused, value));
    lazy.expand(lazy.root());
    auto let = lazy.get(lazy.root());
    require(let.m_num_children == 3, "let constructor order lost");
    require(is_eqp(lazy.get(let.m_children[1]).m_expr, unused), "unused let changed");
    require(!lazy.get(let.m_children[1]).m_expanded, "unused let was traversed");
    require(is_eqp(lazy.get(let.m_children[2]).m_expr, value), "let body changed");

    expr body = mk_bvar(nat(0));
    expr lam = mk_lambda(name("x"), type, body, mk_binder_info());
    expr implicit = mk_lambda(name("x"), type, body, mk_implicit_binder_info());
    reduction_region binders(lam);
    auto other = binders.capture(implicit);
    require(other != binders.root(), "binder annotations were merged");
    binders.expand(other);
    require(is_implicit(binding_info(binders.get(other).m_expr)), "implicit binder lost");
    require(binders.capture(lam) == binders.root(), "identity capture is unstable");

    std::vector<std::pair<expr, std::vector<expr>>> cases{
        {body, {}}, {mk_fvar(name("free")), {}}, {mk_mvar(name("meta")), {}},
        {mk_sort(mk_univ_param(name("u"))), {}},
        {mk_const(name("c"), levels{mk_univ_param(name("u"))}), {}},
        {value, {}}, {mk_lit(literal("unchanged")), {}},
        {mk_app(type, value), {type, value}},
        {lam, {type, body}},
        {mk_pi(name("x"), type, body, binder_info::Implicit), {type, body}},
        {mk_let(name("x"), type, value, body, false), {type, value, body}},
        {mk_mdata(set_nat(kvmap(), name("annotation"), 7), value), {value}},
        {mk_proj(name("Pair"), 1, shared), {shared}}
    };
    std::set<expr_kind> kinds;
    for (auto const & entry : cases) {
        reduction_region arena(entry.first);
        arena.expand(arena.root());
        auto n = arena.get(arena.root());
        kinds.insert(n.m_expr.kind());
        require(is_eqp(n.m_expr, entry.first), "syntax payload was changed");
        require(n.m_num_children == entry.second.size(), "constructor arity changed");
        for (unsigned i = 0; i < n.m_num_children; ++i) {
            require(is_eqp(arena.get(n.m_children[i]).m_expr, entry.second[i]),
                    "constructor child order changed");
            require(!arena.get(n.m_children[i]).m_expanded, "child was traversed eagerly");
        }
    }
    require(kinds.size() == 12, "not every Expr constructor was covered");

    // The arena, not a caller's temporary expression, owns captured syntax.
    reduction_region owned(mk_app(mk_const(name("temporary")), value));
    owned.expand(owned.root());
    require(const_name(owned.get(owned.get(owned.root()).m_children[0]).m_expr)
                == name("temporary"), "temporary root lifetime was not retained");

    // A deep syntax DAG expands one layer at a time without C++ recursion.
    expr deep = value;
    for (unsigned i = 0; i < 20000; ++i) deep = mk_app(deep, value);
    reduction_region deep_graph(deep);
    for (std::size_t i = 0; i < deep_graph.size(); ++i) deep_graph.expand(i);
    require(deep_graph.size() == 20001, "deep graph lost sharing or nodes");

    bool rejected = false;
    try { deep_graph.expand(deep_graph.size()); }
    catch (std::out_of_range const &) { rejected = true; }
    require(rejected, "invalid ID was accepted");
    require(!reduction_mode::full().is_core(), "full mode changed");
    auto mode = reduction_mode::core(true, false);
    require(mode.is_core() && mode.cheap_rec() && !mode.cheap_proj(), "core flags lost");
    std::printf("%u actual-expression capture checks passed\n", checks);
}
