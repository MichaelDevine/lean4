/* Actual Lean expressions and runtime, exercising insert-only storage. */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/expr_whnf_cache.h"

extern "C" void lean_initialize();
using namespace lean;

static unsigned checks = 0;
static void require(bool condition, char const * message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "check failed: %s\n", message);
        std::exit(1);
    }
}

static expr number(unsigned n) { return mk_lit(literal(n)); }
static expr power(unsigned n) {
    return mk_lit(literal(nat(lean_nat_pow(lean_box(2), lean_box(n)))));
}

static expr clone(expr const & e) {
    switch (e.kind()) {
    case expr_kind::BVar: return mk_bvar(bvar_idx(e));
    case expr_kind::FVar: return mk_fvar(fvar_name(e));
    case expr_kind::MVar: return mk_mvar(mvar_name(e));
    case expr_kind::Sort: return mk_sort(sort_level(e));
    case expr_kind::Const: return mk_const(const_name(e), const_levels(e));
    case expr_kind::Lit:
        if (is_nat_lit(e))
            return mk_lit(literal(nat(lit_value(e).get_nat().to_std_string().c_str())));
        return mk_lit(literal(lit_value(e).get_string().data()));
    case expr_kind::App: return mk_app(clone(app_fn(e)), clone(app_arg(e)));
    case expr_kind::Lambda: case expr_kind::Pi:
        return mk_binding(e.kind(), binding_name(e), clone(binding_domain(e)), clone(binding_body(e)), binding_info(e));
    case expr_kind::Let:
        return mk_let(let_name(e), clone(let_type(e)), clone(let_value(e)), clone(let_body(e)), let_nondep(e));
    case expr_kind::MData: return mk_mdata(mdata_data(e), clone(mdata_expr(e)));
    case expr_kind::Proj: return mk_proj(proj_sname(e), proj_idx(e), clone(proj_expr(e)));
    }
    std::abort();
}

static void expect(expr_whnf_cache const & cache, expr const & key, expr const & value) {
    auto found = cache.find(key);
    require(found && *found == value, "missing or incorrect cached result");
}

int main() {
    initialize_runtime_module();
    lean_initialize();
    lean_io_mark_end_initialization();

    expr_whnf_cache cache;
    expr f = mk_const(name("f")), n = mk_const(name("Nat")), big = power(1024);
    expr variable = mk_bvar(0);
    std::vector<expr> examples{
        variable, mk_fvar(name("free")), mk_mvar(name("meta")),
        mk_sort(mk_succ(mk_univ_param(name("u")))),
        mk_const(name("c"), levels{mk_univ_param(name("u"))}),
        mk_app(f, big), mk_lambda(name("x"), n, variable, binder_info::Default),
        mk_pi(name("x"), n, variable, binder_info::Implicit),
        mk_let(name("x"), n, big, variable, false), big,
        mk_mdata(set_nat(kvmap(), name("key"), 7), big),
        mk_proj(name("Prod"), 1, mk_app(f, big)), mk_lit(literal("hello")), expr()};
    for (unsigned i = 0; i < examples.size(); ++i) {
        require(!is_scalar(examples[i].raw()), "valid Expr is scalar");
        require(!cache.find(examples[i]), "empty or absent slot produced result");
        cache.insert(examples[i], number(i));
        expr rebuilt = clone(examples[i]);
        require(rebuilt == examples[i], "clone differs structurally");
        cache.insert(rebuilt, number(999));
        expect(cache, rebuilt, number(i));
    }
    require(cache.size() == examples.size(), "equal key inserted twice");
    expr lam = mk_lambda(name("y"), n, variable, binder_info::Implicit);
    require(lam == examples[6], "binder-insensitivity fixture");
    expect(cache, lam, number(6));

    // Intentional full fingerprint collisions below its sampling depth.
    std::vector<expr> collision_keys;
    for (unsigned i = 64; i < 192; ++i) {
        expr key = power(i);
        for (unsigned j = 0; j < 5; ++j) key = mk_app(f, key);
        if (!collision_keys.empty()) {
            require(expr_cache_hash()(key) == expr_cache_hash()(collision_keys.front()), "collision fixture hash differs");
            require(key != collision_keys.front(), "collision fixture keys equal");
        }
        cache.insert(key, number(i));
        collision_keys.push_back(key);
    }
    for (unsigned i = 0; i < collision_keys.size(); ++i)
        expect(cache, clone(collision_keys[i]), number(i + 64));

    // Thousands of independent keys, with several table growths and misses.
    for (unsigned i = 0; i < 20000; ++i) cache.insert(mk_app(f, number(i)), number(i + 1));
    for (unsigned i = 0; i < 20000; ++i) expect(cache, mk_app(clone(f), number(i)), number(i + 1));
    require(!cache.find(mk_app(f, number(20001))), "absent key accepted");
    for (unsigned i = 0; i < collision_keys.size(); ++i) expect(cache, collision_keys[i], number(i + 64));

    expr_whnf_cache copied(cache);
    copied.insert(number(70000), number(19));
    require(!cache.find(number(70000)), "copy mutates original");
    expr_whnf_cache moved(std::move(copied));
    require(copied.size() == 0 && !copied.find(number(70000)), "moved-from map is not empty");
    copied.insert(number(1), number(3));
    expect(copied, number(1), number(3));
    expect(moved, number(70000), number(19));
    auto & self = moved;
    moved = self;
    moved = std::move(self);
    expect(moved, number(70000), number(19));
    copied = cache;
    expect(copied, clone(collision_keys.back()), number(191));
    copied = expr_whnf_cache();
    require(copied.size() == 0 && !copied.find(collision_keys.back()), "assignment retains entries");

    // A borrowed result is both the new key and value across mandatory growth.
    expr_whnf_cache aliases;
    for (unsigned i = 0; i < 12; ++i) aliases.insert(number(i), number(i + 100));
    auto borrowed = aliases.find(number(0));
    require(borrowed != nullptr, "alias fixture missing");
    aliases.insert(*borrowed, *borrowed);
    expect(aliases, number(100), number(100));
    require(aliases.size() == 13, "alias insertion lost across growth");

    expr_whnf_cache retained;
    { expr key = mk_app(f, power(2048)), value = mk_app(f, power(4096)); retained.insert(key, value); }
    for (unsigned i = 0; i < 1000; ++i) { expr churn = mk_app(f, power(128 + i)); }
    expect(retained, mk_app(f, power(2048)), mk_app(f, power(4096)));
    // Count ownership of actual nonpersistent, single-threaded Expr roots.
    // Growth moves handles, copies retain once, and destruction releases once.
    expr counted_key = mk_app(f, power(2031)), counted_value = mk_app(f, power(4079));
    int key_rc = counted_key.raw()->m_rc, value_rc = counted_value.raw()->m_rc;
    require(key_rc > 0 && value_rc > 0, "reference-count fixture is persistent or shared-threaded");
    {
        expr_whnf_cache owners;
        owners.insert(counted_key, counted_value);
        require(counted_key.raw()->m_rc == key_rc + 1 && counted_value.raw()->m_rc == value_rc + 1,
                "insertion does not retain exactly one key and value");
        for (unsigned i = 0; i < 1000; ++i) owners.insert(number(i), number(i + 1));
        require(counted_key.raw()->m_rc == key_rc + 1 && counted_value.raw()->m_rc == value_rc + 1,
                "growth leaks or drops references");
        {
            expr_whnf_cache copy(owners);
            expr_whnf_cache transferred(std::move(copy));
            require(counted_key.raw()->m_rc == key_rc + 2 && counted_value.raw()->m_rc == value_rc + 2,
                    "copy/move ownership differs");
        }
        require(counted_key.raw()->m_rc == key_rc + 1 && counted_value.raw()->m_rc == value_rc + 1,
                "copied cache destruction leaks references");
    }
    require(counted_key.raw()->m_rc == key_rc && counted_value.raw()->m_rc == value_rc,
            "cache destruction leaks references");
    std::printf("{\"valid\":true,\"checks\":%u,\"expr_kinds\":12,\"collision_keys\":128}\n", checks);
}
