/* Actual Lean expressions, equality and allocator; no expression-model stub. */
#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/expr_cache_hash.h"
#include "kernel/expr_maps.h"

extern "C" void lean_initialize();

using namespace lean;
using cache = lean::unordered_map<expr, unsigned, expr_cache_hash, std::equal_to<expr>>;

static unsigned checks = 0;
static void require(bool condition, char const * message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "check failed: %s\n", message);
        std::exit(1);
    }
}

static expr power_lit(unsigned exponent) {
    return mk_lit(literal(nat(lean_nat_pow(lean_box(2), lean_box(exponent)))));
}

/* Reconstruct every Expr node. Names and universe levels need not be rebuilt:
   their equality is already structural, and the fingerprint uses stored hash. */
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

static void equal_keys(expr const & a, expr const & b) {
    require(a == b, "expected original structural equality");
    require(expr_cache_hash()(a) == expr_cache_hash()(b), "equal keys have different fingerprints");
    cache entries;
    entries.emplace(a, 17);
    entries.emplace(b, 29);
    require(entries.size() == 1, "equal cache keys were duplicated");
    require(entries.at(b) == 17, "reconstructed key did not retrieve original result");
    entries.rehash(101);
    require(entries.at(a) == 17 && entries.at(b) == 17, "rehash changed an equal-key result");
}

static void distinct_keys(expr const & a, expr const & b) {
    require(a != b, "expected original structural inequality");
    cache entries;
    entries.emplace(a, 17);
    entries.emplace(b, 29);
    require(entries.size() == 2, "distinct cache keys were merged");
    require(entries.at(clone(a)) == 17 && entries.at(clone(b)) == 29, "distinct result lookup failed");
    entries.rehash(101);
    require(entries.at(a) == 17 && entries.at(b) == 29, "rehash changed distinct results");
}

int main() {
    initialize_runtime_module();
    lean_initialize();
    lean_io_mark_end_initialization();

    static_assert(std::is_empty<expr_cache_hash>::value, "fingerprint must be stateless");
    static_assert(!std::is_nothrow_invocable<expr_cache_hash, expr const &>::value,
                  "preserve cached node hashes on libstdc++");
    static_assert(std::is_same<cache::key_equal, std::equal_to<expr>>::value,
                  "retain the original exact equality");
    static_assert(std::is_same<cache::allocator_type, expr_map<unsigned>::allocator_type>::value,
                  "retain the Lean allocator");
#ifdef __GLIBCXX__
    static_assert(std::__cache_default<expr, expr_cache_hash>::value, "node hashes are not cached");
    static_assert(std::__is_fast_hash<expr_cache_hash>::value, "unexpected small-table linear scans");
#endif

    expr n = mk_const(name("Nat"));
    expr f = mk_const(name("f"));
    expr big = power_lit(1024);
    expr variable = mk_bvar(0);
    std::vector<expr> examples{
        variable, mk_fvar(name("free")), mk_mvar(name("meta")),
        mk_sort(mk_succ(mk_univ_param(name("u")))),
        mk_const(name("c"), levels{mk_univ_param(name("u"))}),
        mk_app(f, big), mk_lambda(name("x"), n, variable, binder_info::Default),
        mk_pi(name("x"), n, variable, binder_info::Implicit),
        mk_let(name("x"), n, big, variable, false), big,
        mk_mdata(set_nat(kvmap(), name("key"), 7), big),
        mk_proj(name("Prod"), 1, mk_app(f, big)), mk_lit(literal("hello"))};
    std::set<unsigned> kinds;
    for (expr const & e : examples) {
        kinds.insert(static_cast<unsigned>(e.kind()));
        expr rebuilt = clone(e);
        require(!is_eqp(e, rebuilt), "clone retained root identity");
        equal_keys(e, rebuilt);
        equal_keys(mk_app(f, e), mk_app(clone(f), rebuilt));
    }
    require(kinds.size() == 12, "not every Expr kind was tested");

    for (binder_info info : {binder_info::Default, binder_info::Implicit,
                            binder_info::StrictImplicit, binder_info::InstImplicit, binder_info::Rec}) {
        equal_keys(mk_lambda(name("x"), n, variable, binder_info::Default),
                   mk_lambda(name("renamed"), clone(n), clone(variable), info));
        equal_keys(mk_pi(name("x"), n, variable, binder_info::Default),
                   mk_pi(name("renamed"), clone(n), clone(variable), info));
    }
    equal_keys(mk_let(name("x"), n, big, variable, false),
               mk_let(name("renamed"), clone(n), clone(big), clone(variable), false));
    expr dependent = mk_let(name("x"), n, big, variable, false);
    expr nondependent = mk_let(name("x"), n, big, variable, true);
    require(expr_cache_hash()(dependent) == expr_cache_hash()(nondependent), "nondep collision control disappeared");
    distinct_keys(dependent, nondependent);
    expr md1 = mk_mdata(set_nat(kvmap(), name("key"), 7), big);
    expr md2 = mk_mdata(set_nat(kvmap(), name("key"), 8), clone(big));
    require(expr_cache_hash()(md1) == expr_cache_hash()(md2), "metadata collision control disappeared");
    distinct_keys(md1, md2);
    distinct_keys(mk_proj(name("Prod"), 0, big), mk_proj(name("Prod"), 1, big));
    distinct_keys(mk_proj(name("Prod"), 0, big), mk_proj(name("Other"), 0, big));
    distinct_keys(mk_const(name("c"), levels{mk_level_zero()}),
                  mk_const(name("c"), levels{mk_level_one()}));
    distinct_keys(mk_bvar(0), mk_bvar(1));
    distinct_keys(mk_fvar(name("free")), mk_fvar(name("other")));
    distinct_keys(mk_mvar(name("meta")), mk_mvar(name("other")));
    distinct_keys(mk_sort(mk_level_zero()), mk_sort(mk_level_one()));
    distinct_keys(mk_const(name("c")), mk_const(name("other")));
    distinct_keys(mk_lit(literal("hello")), mk_lit(literal("world")));
    distinct_keys(mk_lit(literal("0")), mk_lit(literal(0u)));

    expr shared = mk_app(big, big);
    expr unshared = mk_app(clone(big), clone(big));
    require(is_eqp(app_fn(shared), app_arg(shared)), "shared DAG fixture is not shared");
    require(!is_eqp(app_fn(unshared), app_arg(unshared)), "unshared DAG fixture is shared");
    equal_keys(shared, unshared);
    size_t original_fingerprint = expr_cache_hash()(big);
    { std::vector<expr> owners(100, big); require(expr_cache_hash()(big) == original_fingerprint, "refcount changed fingerprint"); }
    require(expr_cache_hash()(big) == original_fingerprint, "released owners changed fingerprint");

    std::set<size_t> full_hashes;
    for (unsigned exponent : {64u, 65u, 127u, 128u, 191u, 192u, 255u, 256u, 1024u, 4096u}) {
        expr p = power_lit(exponent);
        require(hash(p) == hash(power_lit(64)), "old collision family changed");
        require(full_hashes.insert(expr_cache_hash()(p)).second, "full-limb power hashes collide");
        equal_keys(p, clone(p));
    }
    for (unsigned exponent : {0u, 1u, 31u, 32u, 62u, 63u, 64u, 1024u}) {
        nat p(lean_nat_pow(lean_box(2), lean_box(exponent)));
        for (nat const & value : {p - nat(1u), p, p + nat(1u)}) {
            expr literal = mk_lit(lean::literal(value));
            equal_keys(literal, clone(literal));
        }
    }
    nat high(lean_nat_pow(lean_box(2), lean_box(1024)));
    nat middle(lean_nat_pow(lean_box(2), lean_box(512)));
    expr h1 = mk_lit(literal(high + nat(3u)));
    expr h2 = mk_lit(literal(high + middle + nat(3u)));
    require(hash(h1) == hash(h2), "middle-limb fixture does not share old hash");
    require(expr_cache_hash()(h1) != expr_cache_hash()(h2), "middle limb was ignored");
    distinct_keys(h1, h2);

    expr p64 = power_lit(64), p65 = power_lit(65);
    expr left1 = mk_app(mk_app(f, p64), n), left2 = mk_app(mk_app(f, p65), n);
    require(expr_cache_hash()(left1) != expr_cache_hash()(left2), "boundary Nat operand was not fingerprinted");
    for (unsigned i = 0; i < 5; ++i) { p64 = mk_app(f, p64); p65 = mk_app(f, p65); }
    require(hash(p64) == hash(p65), "deep collision fixture lost old collision");
    require(expr_cache_hash()(p64) == expr_cache_hash()(p65), "residual fingerprint collision fixture failed");
    distinct_keys(p64, p65);

    /* A surviving map owns the key after the caller releases its expression. */
    cache retained;
    { expr transient = mk_app(f, power_lit(2048)); retained.emplace(transient, 41); }
    for (unsigned i = 0; i < 1000; ++i) { expr transient = mk_app(f, mk_lit(literal(i))); }
    require(retained.at(mk_app(clone(f), power_lit(2048))) == 41, "cache lost ownership of its key");
    retained.clear();
    require(retained.empty(), "cache clear retained an entry");

    std::printf("{\"checks\":%u,\"expr_kinds\":%zu,\"power_hashes\":%zu,\"valid\":true}\n",
                checks, kinds.size(), full_hashes.size());
    return 0;
}
