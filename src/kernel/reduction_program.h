/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <vector>
#include "kernel/reduction_region.h"
#include "kernel/reduction_machine.h"

namespace lean {

/** Compile the supported lazy syntax fragment, leaving other nodes as
    explicit boundaries. Domains, annotations and exact literal payloads
    remain owned by the source arena; they are not numerically interpreted
    by the closure machine. This is not yet a complete device reduction. */
inline std::vector<reduction::instruction> make_reduction_program(reduction_region & region) {
    using namespace reduction;
    std::vector<instruction> code;
    std::vector<reduction_region::node_id> todo{region.root()};
    std::vector<bool> visited;
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
        case expr_kind::Lit: case expr_kind::Sort: case expr_kind::Pi:
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
        case expr_kind::Const: case expr_kind::FVar: case expr_kind::MVar: case expr_kind::Proj:
            break;
        }
        if (code.size() < region.size())
            code.resize(region.size(), {opcode::unsupported, 0, 0, 0});
        code[id] = n;
    }
    return code;
}
}
