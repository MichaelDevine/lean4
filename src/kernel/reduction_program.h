/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <vector>
#include "kernel/reduction_region.h"
#include "kernel/reduction_machine.h"
#include "kernel/reduction_literal.h"
#include "kernel/reduction_primitive.h"
#include "kernel/instantiate.h"
#include "kernel/expr_maps.h"

namespace lean {

/** Lower supported syntax without evaluating it. Source domains/annotations
    remain owned by the arena. Full-mode numeric literals also get exact limb
    storage, and supported primitives are classified against the supplied
    immutable environment during lowering. Without that snapshot, primitives
    remain boundaries. Full-mode definition bodies are instantiated/captured,
    not evaluated during lowering; the machine unfolds them only on demand.
    General recursor and projection evaluation remains unsupported. */
inline std::vector<reduction::instruction> make_reduction_program(
    reduction_region & region, std::vector<reduction::natural_limb> & literals,
    reduction_mode mode, environment const * env = nullptr) {
    using namespace reduction;
    std::vector<instruction> code;
    std::vector<reduction_region::node_id> todo{region.root()};
    std::vector<bool> visited;
    // Exact constant/universe instances share captured immutable code, not
    // evaluated results. This cache is local to one environment snapshot.
    expr_map<reduction_region::node_id> definitions;
    expr nat_zero, nat_add, nat_sub, nat_mul;
    bool zero_supported = false, add_supported = false, sub_supported = false, mul_supported = false;
    if (!mode.is_core()) {
        nat_zero = mk_const(name({"Nat", "zero"}));
        nat_add = mk_const(name({"Nat", "add"}));
        nat_sub = mk_const(name({"Nat", "sub"}));
        nat_mul = mk_const(name({"Nat", "mul"}));
        // Core iota reduction precedes primitive arithmetic in the reference
        // checker. Do not bypass an environment recursor under a builtin name.
        // Bare/operand Nat.zero also undergoes full delta reduction first.
        auto supported = [&](expr const & e, bool zero) {
            if (env == nullptr) return false;
            auto info = env->find(const_name(e));
            return !info || (!info->is_recursor() && (!zero || !info->has_value()));
        };
        zero_supported = supported(nat_zero, true);
        add_supported = supported(nat_add, false);
        sub_supported = supported(nat_sub, false);
        mul_supported = supported(nat_mul, false);
    }
    while (!todo.empty()) {
        auto id = todo.back();
        todo.pop_back();
        if (visited.size() < region.size()) visited.resize(region.size());
        if (visited[id]) continue;
        visited[id] = true;
        expr e = region.get(id).m_expr;
        instruction n{opcode::unsupported, 0, 0, id};
        auto expand = [&]() { region.expand(id); };
        auto child = [&](unsigned i) {
            auto c = region.get(id).m_children[i];
            todo.push_back(c);
            return static_cast<index>(c);
        };
        switch (e.kind()) {
        case expr_kind::Lit:
            n.m_op = opcode::value;
            if (!mode.is_core() && is_nat_lit(e)) {
                auto value = append_reduction_literal(lit_value(e).get_nat(), literals);
                n.m_op = opcode::natural; n.m_first = value.m_offset; n.m_second = value.m_size;
            }
            break;
        case expr_kind::Sort: case expr_kind::Pi:
            n.m_op = opcode::value;
            break;
        case expr_kind::BVar:
            if (bvar_idx(e).is_small()) {
                n.m_op = opcode::bound;
                n.m_first = bvar_idx(e).get_small_value();
            }
            break;
        case expr_kind::App:
            expand(); n.m_op = opcode::app; n.m_first = child(0); n.m_second = child(1);
            break;
        case expr_kind::Lambda:
            expand(); n.m_op = opcode::lambda; n.m_first = child(1);
            break;
        case expr_kind::Let:
            expand(); n.m_op = opcode::let; n.m_first = child(1); n.m_second = child(2);
            break;
        case expr_kind::MData:
            expand(); n.m_op = opcode::alias; n.m_first = child(0);
            break;
        case expr_kind::Const:
            // Exact constants (including their universe arguments), not a
            // name-only match. Core mode must not acquire full Nat reduction.
            if (!mode.is_core()) {
                if (zero_supported && e == nat_zero) n.m_op = opcode::natural_zero;
                if (add_supported && e == nat_add) n.m_op = opcode::natural_add;
                if (sub_supported && e == nat_sub) n.m_op = opcode::natural_subtract;
                if (mul_supported && e == nat_mul) n.m_op = opcode::natural_multiply;
                if (n.m_op == opcode::unsupported && env != nullptr &&
                    classify_reduction_primitive(e, 1) == reduction_primitive::none &&
                    classify_reduction_primitive(e, 2) == reduction_primitive::none) {
                    auto info = env->find(const_name(e));
                    // Match is_delta/unfold_definition_core exactly: opaque
                    // bodies are excluded; universe arity must be correct.
                    if (info && info->has_value() && length(const_levels(e)) == info->get_num_lparams()) {
                        auto it = definitions.find(e);
                        reduction_region::node_id body;
                        if (it != definitions.end()) body = it->second;
                        else {
                            body = region.capture(instantiate_value_lparams(*info, const_levels(e)));
                            definitions.emplace(e, body);
                        }
                        todo.push_back(body);
                        n.m_op = opcode::definition; n.m_first = body;
                    }
                }
            }
            break;
        case expr_kind::FVar: case expr_kind::MVar: case expr_kind::Proj:
            break;
        }
        if (code.size() < region.size())
            code.resize(region.size(), {opcode::unsupported, 0, 0, 0});
        code[id] = n;
    }
    return code;
}

/** The original syntax-only packing entry point remains useful to callers
    which intentionally do not provide numeric storage or full reduction. */
inline std::vector<reduction::instruction> make_reduction_program(reduction_region & region) {
    std::vector<reduction::natural_limb> literals;
    return make_reduction_program(region, literals, reduction_mode::core(false, false));
}
}
