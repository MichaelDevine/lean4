/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "kernel/environment.h"
#include "kernel/local_ctx.h"

namespace lean {

/** Host-owned expression arena for a demanded reduction.

    Capturing or expanding a node does not reduce it. In particular, a captured
    let value, application argument or recursor alternative remains unevaluated.
    IDs identify syntax, not evaluated closures: an evaluator must also carry
    the binding environment and reduction mode when memoizing a result.

    This is the host capture boundary, not a device wire format. `expr` references
    and the identity index must never be dereferenced on a device. Expansion is
    single-owner; callers must not retain node references across insertions. */
class reduction_region {
public:
    using node_id = std::size_t;

    struct node {
        expr                   m_expr;
        std::array<node_id, 3>  m_children{};
        unsigned               m_num_children = 0;
        bool                   m_expanded = false;

        explicit node(expr const & e):m_expr(e) {}
    };

private:
    std::vector<node> m_nodes;
    std::unordered_map<object *, node_id> m_ids;
    node_id m_root;

public:
    explicit reduction_region(expr const & root):m_root(capture(root)) {}

    node_id root() const { return m_root; }
    std::size_t size() const { return m_nodes.size(); }
    node const & get(node_id id) const { return m_nodes.at(id); }

    /** Retain the exact expression, preserving object-identity sharing.
        Do not use structural equality here: kernel expression equality can
        intentionally ignore annotations which capture must retain. */
    node_id capture(expr const & e) {
        object * key = e.raw();
        auto it = m_ids.find(key);
        if (it != m_ids.end()) return it->second;
        node_id id = m_nodes.size();
        m_nodes.emplace_back(e);
        try {
            m_ids.emplace(key, id);
        } catch (...) {
            m_nodes.pop_back();
            throw;
        }
        return id;
    }

    /** Expose one layer of syntax without traversing or executing children.
        Child positions are the original constructor order: function/argument,
        domain/body, or type/value/body. No host recursion is used. */
    void expand(node_id id) {
        if (get(id).m_expanded) return;
        // Keep an owning copy: capture may relocate m_nodes, and its argument
        // must not be a reference into the relocated vector.
        expr e = get(id).m_expr;
        std::array<node_id, 3> children{};
        unsigned count = 0;
        auto add = [&](expr const & child) { children[count++] = capture(child); };
        switch (e.kind()) {
        case expr_kind::BVar: case expr_kind::FVar: case expr_kind::MVar:
        case expr_kind::Sort: case expr_kind::Const: case expr_kind::Lit:
            break;
        case expr_kind::App:
            add(app_fn(e)); add(app_arg(e));
            break;
        case expr_kind::Lambda: case expr_kind::Pi:
            add(binding_domain(e)); add(binding_body(e));
            break;
        case expr_kind::Let:
            add(let_type(e)); add(let_value(e)); add(let_body(e));
            break;
        case expr_kind::MData:
            add(mdata_expr(e));
            break;
        case expr_kind::Proj:
            add(proj_expr(e));
            break;
        }
        // If allocation failed, the parent stays unexpanded and retryable.
        // Already retained children remain valid, reusable arena entries.
        node & n = m_nodes.at(id);
        n.m_children = children;
        n.m_num_children = count;
        n.m_expanded = true;
    }
};

/** Full and core reduction are distinct requests. The cheap flags apply only
    to the core request, matching the existing type-checker interface. */
class reduction_mode {
    bool m_core;
    bool m_cheap_rec;
    bool m_cheap_proj;
    reduction_mode(bool core, bool cheap_rec, bool cheap_proj):
        m_core(core), m_cheap_rec(cheap_rec), m_cheap_proj(cheap_proj) {}
public:
    static reduction_mode full() { return reduction_mode(false, false, false); }
    static reduction_mode core(bool cheap_rec, bool cheap_proj) {
        return reduction_mode(true, cheap_rec, cheap_proj);
    }
    bool is_core() const { return m_core; }
    bool cheap_rec() const { return m_cheap_rec; }
    bool cheap_proj() const { return m_cheap_proj; }
};

/** Retain the semantic snapshot separately from any mutable checker caches.
    A backend must still capture/marshal the demanded definitions and locals;
    possession of this snapshot is not evidence of device eligibility. */
class reduction_request {
    environment      m_env;
    local_ctx        m_lctx;
    reduction_mode   m_mode;
    reduction_region m_region;
public:
    reduction_request(environment const & env, local_ctx const & lctx,
                      reduction_mode mode, expr const & root):
        m_env(env), m_lctx(lctx), m_mode(mode), m_region(root) {}

    environment const & env() const { return m_env; }
    local_ctx const & lctx() const { return m_lctx; }
    reduction_mode mode() const { return m_mode; }
    reduction_region & region() { return m_region; }
    reduction_region const & region() const { return m_region; }
};
}
