/* Additive demand-preserving, environment-bound definition capture controls.
   The reference variant uses only the original checker, without our hook. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#ifndef LEAN_REDUCTION_DEFINITIONS_REFERENCE
#include "library/closure_reduction_extension.h"
#endif
#ifdef LEAN_REDUCTION_DEFINITIONS_GPU
#include "library/reduction_sycl.h"
#endif
extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);
using namespace lean;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static expr binary(char const * op, expr const & a, expr const & b) {
    return mk_app(mk_app(mk_const(name({"Nat", op})), a), b);
}
static expr lambda(expr const & type, expr const & body) {
    return mk_lambda(name("x"), type, body, binder_info::Default);
}
static expr arrow(expr const & domain, expr const & codomain) {
    return mk_pi(name("x"), domain, codomain, binder_info::Default);
}

#ifndef LEAN_REDUCTION_DEFINITIONS_REFERENCE
using namespace lean::reduction;
template<typename Backend> struct counted_backend {
    Backend & m_backend;
    std::size_t m_calls = 0, m_definitions = 0, m_bodies = 0;
    outcome operator()(std::vector<instruction> const & code, machine & state,
                       std::vector<closure> & arguments, std::vector<binding> & bindings,
                       arithmetic_workspace & arithmetic) {
        ++m_calls;
        m_definitions = 0;
        std::set<reduction::index> bodies;
        for (auto const & n : code) if (n.m_op == opcode::definition) { ++m_definitions; bodies.insert(n.m_first); }
        m_bodies = bodies.size();
        return m_backend(code, state, arguments, bindings, arithmetic);
    }
};

static optional<reduction_workspace> provision(reduction_session const & s, outcome reason) {
    // Fixture capacities, including the known 64-step arithmetic body. This
    // is not a runtime admission policy or a termination/workspace ceiling.
    if (reason == outcome::running) {
        auto size = s.program_size();
        return optional<reduction_workspace>({2 * size, 2 * size, 2 * size,
                                              4 * size * (s.literal_size() + 256), 4 * (s.literal_size() + 256)});
    }
    return optional<reduction_workspace>({s.argument_capacity() + (reason == outcome::need_arguments),
        s.binding_capacity() + (reason == outcome::need_bindings),
        s.frame_capacity() + (reason == outcome::need_frames),
        std::max<reduction::index>(s.value_capacity(), s.state().m_required_values),
        std::max<reduction::index>(s.column_capacity(), s.state().m_required_columns)});
}
#endif

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment original(lean_elab_environment_to_kernel_env(raw)), env = original;
    expr sort0 = mk_sort(mk_level_zero()), sort1 = mk_sort(mk_level_one());
    expr sort2 = mk_sort(mk_succ(mk_level_one()));
    auto add = [&](name const & n, names const & parameters, expr const & type, expr const & value) {
        env = env.add(mk_definition(n, parameters, type, value, reducibility_hints::mk_abbreviation()));
    };
    name id("regionIdentity"), alias("regionAlias"), choose("regionFirst"), poly("regionUniverse");
    add(id, names(), arrow(sort1, sort1), lambda(sort1, mk_bvar(0)));
    add(alias, names(), arrow(sort1, sort1), mk_const(id));
    add(choose, names(), arrow(sort1, arrow(sort1, sort1)), lambda(sort1, lambda(sort1, mk_bvar(1))));
    name u("u");
    expr sortu = mk_sort(mk_univ_param(u));
    add(poly, names{u}, arrow(sortu, sortu), lambda(sortu, mk_bvar(0)));
    name opaque("regionOpaque");
    env = env.add(mk_opaque(opaque, names(), sort1, sort0, false));
    name p("RegionP"), theorem("regionProofIdentity");
    env = env.add(mk_axiom(p, names(), sort0));
    expr prop = mk_const(p);
    env = env.add(mk_theorem(theorem, names(), arrow(prop, prop), lambda(prop, mk_bvar(0))));

    // Checked synthetic environment for numerical controls. It is not a
    // published arithmetic development or a claim of a new proof theorem.
    expr nat_type = mk_const(name("Nat"));
    env = env.add(mk_axiom(name("Nat"), names(), sort1));
    for (auto op : {"add", "mul"})
        env = env.add(mk_axiom(name({"Nat", op}), names(), arrow(nat_type, arrow(nat_type, nat_type))));
    expr one = mk_lit(literal(1u)), seven = mk_lit(literal(7u));
    name step("regionArithmeticStep"), chain("regionArithmeticChain");
    add(step, names(), arrow(nat_type, nat_type),
        lambda(nat_type, binary("add", binary("mul", mk_bvar(0), seven), one)));
    expr body = mk_bvar(0);
    for (unsigned i = 0; i < 64; ++i) body = mk_app(mk_const(step), body);
    add(chain, names(), arrow(nat_type, nat_type), lambda(nat_type, body));
    expr numeric = mk_app(mk_const(chain), mk_lit(literal(nat("340282366920938463463374607431768211455"))));

    std::vector<expr> cases{
        mk_app(mk_const(id), sort0), mk_app(mk_const(alias), sort0),
        mk_app(mk_app(mk_const(choose), sort0), mk_const(opaque)),
        mk_app(mk_const(choose), sort0), mk_const(alias), mk_const(theorem),
        mk_app(mk_const(poly, levels{mk_level_one()}), sort0),
        mk_app(mk_const(poly, levels{mk_succ(mk_level_one())}), sort1),
        mk_let(name("local"), sort1, sort0, mk_app(mk_const(alias), mk_bvar(0)), false),
        mk_app(mk_const(step), seven), numeric
    };
    std::vector<expr> expected;
    for (auto const & input : cases) {
        type_checker checker(env);
        checker.check(input);
        expected.push_back(checker.whnf(input));
    }
    require(expected[0] == sort0 && expected[1] == sort0 && expected[2] == sort0 &&
            expected[6] == sort0 && expected[7] == sort1 && expected[8] == sort0 &&
            expected[9] == mk_lit(literal(50u)), "ordinary definition fixtures have unexpected normal forms");
    nat manual("340282366920938463463374607431768211455");
    for (unsigned i = 0; i < 64; ++i) manual = manual * nat(7u) + nat(1u);
    require(expected.back() == mk_lit(literal(manual)), "ordinary arithmetic definition fixture differs from Nat arithmetic");

    // Exercise every ordinary Nat dispatch arm in both the unmodified
    // reference executable and the modified checker, with fixed answers.
    struct primitive_case { char const * m_name; unsigned m_expected; };
    primitive_case primitives[]{{"add", 17}, {"sub", 7}, {"mul", 60}, {"pow", 248832},
        {"gcd", 1}, {"mod", 2}, {"div", 2}, {"land", 4}, {"lor", 13}, {"xor", 9},
        {"shiftLeft", 384}, {"shiftRight", 0}};
    for (auto const & primitive : primitives) {
        type_checker checker(env);
        require(checker.whnf(binary(primitive.m_name, mk_lit(literal(12u)), mk_lit(literal(5u)))) ==
                mk_lit(literal(primitive.m_expected)), "ordinary Nat dispatch changed an operation");
    }
    for (auto op : {"beq", "ble"}) {
        type_checker checker(env);
        require(checker.whnf(binary(op, seven, one)) == mk_const(name({"Bool", "false"})) &&
                checker.whnf(binary(op, seven, seven)) == mk_const(name({"Bool", "true"})),
                "ordinary Nat predicate dispatch changed a result");
    }
    type_checker successor(env);
    require(successor.whnf(mk_app(mk_const(name({"Nat", "succ"})), seven)) == mk_lit(literal(8u)),
            "ordinary successor dispatch changed the result");

#ifndef LEAN_REDUCTION_DEFINITIONS_REFERENCE
    for (auto const & primitive : primitives) {
        require(classify_reduction_primitive(mk_const(name({"Nat", primitive.m_name})), 2) != reduction_primitive::none &&
                classify_reduction_primitive(mk_const(name({"Nat", primitive.m_name})), 1) == reduction_primitive::none &&
                classify_reduction_primitive(mk_const(name({"Nat", primitive.m_name}), levels{mk_level_one()}), 2) ==
                    reduction_primitive::none, "primitive classification ignored exact arity or universes");
    }
#ifdef LEAN_REDUCTION_DEFINITIONS_GPU
    reduction_sycl_backend device;
#else
    reduction_cpu_backend device;
#endif
    counted_backend backend{device};
    auto admission = provision;
    closure_reduction_extension extension(backend, admission);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        backend.m_calls = 0;
        auto result = extension.reduce(env, local_ctx(), cases[i], reduction_mode::full());
        require(result && is_bi_equal(*result, expected[i]) && extension.last_attempt() == reduction_attempt::complete,
                "captured definition did not match the original full reducer");
        require(backend.m_calls == 1 && backend.m_definitions != 0,
                "definition body was not executed in one region submission");
        if (i + 1 == cases.size())
            require(backend.m_bodies == 2 && backend.m_definitions > backend.m_bodies,
                    "identical constant instances did not share their two captured bodies");
        type_checker accelerated(env);
        {
            scope_reduction_extension enable(&extension);
            require(is_bi_equal(accelerated.whnf(cases[i]), expected[i]), "actual checker definition hook changed the result");
            accelerated.check(cases[i]);
        }
    }
#ifdef LEAN_REDUCTION_DEFINITIONS_GPU
    require(device.work_group_size() > 1, "definition test did not use the cooperative GPU backend");
#endif
    for (unsigned flags = 0; flags < 4; ++flags) {
        auto mode = reduction_mode::core(flags & 1, flags & 2);
        require(!extension.reduce(env, local_ctx(), cases[0], mode), "core mode unfolded a definition");
        type_checker original_core(env), accelerated_core(env);
        auto baseline = original_core.whnf_core(cases[0], flags & 1, flags & 2);
        scope_reduction_extension enable(&extension);
        require(is_bi_equal(accelerated_core.whnf_core(cases[0], flags & 1, flags & 2), baseline),
                "core-mode fallback changed definition semantics");
    }
    for (expr input : {mk_const(opaque), mk_const(poly), mk_const(poly, levels{mk_level_zero(), mk_level_one()})})
        require(!extension.reduce(env, local_ctx(), input, reduction_mode::full()),
                "opaque body or malformed universe instantiation was unfolded");

    // Captured unused arguments may be unsupported or divergent syntax. They
    // are neither normalized by capture nor executed on the device.
    expr self = lambda(sort1, mk_app(mk_bvar(0), mk_bvar(0)));
    expr lazy = mk_app(mk_app(mk_const(choose), sort0), mk_app(self, self));
    auto lazy_result = extension.reduce(env, local_ctx(), lazy, reduction_mode::full());
    require(lazy_result && *lazy_result == sort0, "an unused divergent definition argument was demanded");

    // Same name, different immutable environment: never reuse an old body.
    auto other = original.add(mk_definition(id, names(), arrow(sort2, sort2), lambda(sort2, sort1),
                                           reducibility_hints::mk_abbreviation()));
    auto other_result = extension.reduce(other, local_ctx(), mk_app(mk_const(id), sort1), reduction_mode::full());
    require(other_result && *other_result == sort1, "definition cache crossed environment snapshots");

    reduction_session resumed(numeric, 0, 0, reduction_mode::full(), &env);
    bool needed_bindings = false, needed_values = false;
    while (true) {
        auto status = resumed.advance(device);
        if (status == outcome::complete) break;
        require(status == outcome::need_arguments || status == outcome::need_bindings ||
                status == outcome::need_frames || status == outcome::need_values || status == outcome::need_columns,
                "definition continuation returned an unexpected outcome");
        needed_bindings |= status == outcome::need_bindings;
        needed_values |= status == outcome::need_values;
        auto next = provision(resumed, status);
        resumed.resize_workspace(next->m_arguments, next->m_bindings, next->m_frames, next->m_values, next->m_columns);
    }
    require(needed_bindings && needed_values && is_bi_equal(resumed.reconstruct(), expected.back()),
            "definition workspace continuation changed the result");

    // Native arithmetic must win over a same-named declaration body. This
    // checked shadow definition would incorrectly return zero if captured as
    // ordinary delta code. The real primitive returns floor(10 / 3) = 3.
    auto shadow = env.add(mk_definition(name({"Nat", "div"}), names(), arrow(nat_type, arrow(nat_type, nat_type)),
        lambda(nat_type, lambda(nat_type, mk_lit(literal(0u)))), reducibility_hints::mk_abbreviation()));
    expr division = binary("div", mk_lit(literal(10u)), mk_lit(literal(3u)));
    type_checker shadow_cpu(shadow);
    require(shadow_cpu.whnf(division) == mk_lit(literal(3u)), "primitive-precedence fixture failed");
    require(!extension.reduce(shadow, local_ctx(), division, reduction_mode::full()),
            "unsupported primitive was incorrectly replaced by a declaration body");
    for (auto op : {"reduceNat", "reduceBool"}) {
        name n({"Lean", op});
        auto native_shadow = original.add(mk_definition(n, names(), arrow(sort1, sort1), lambda(sort1, mk_bvar(0)),
                                                        reducibility_hints::mk_abbreviation()));
        require(!extension.reduce(native_shadow, local_ctx(), mk_app(mk_const(n), sort0), reduction_mode::full()),
                "native execution boundary was unfolded as ordinary code");
    }
#endif
    std::puts("definition fixtures passed: 11; arithmetic_steps=64");
}
