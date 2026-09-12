/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kernel/instantiate.h"
#include "kernel/reduction_program.h"

namespace lean {

/** Reconstruct syntax from the closure fragment using the kernel's existing
    capture-avoiding substitution, not normalization. This does not validate
    the computation which produced a machine state or grant proof authority.
    The source arena and used binding prefix must remain unchanged throughout.
    Code IDs are the source-arena IDs assigned by make_reduction_program. */
class reduction_reifier {
    using index = reduction::index;
    using closure = reduction::closure;
    using key = std::pair<index, index>;
    struct key_hash {
        std::size_t operator()(key const & k) const {
            auto a = std::hash<index>()(k.first);
            auto b = std::hash<index>()(k.second);
            return a ^ (b + (a << 6) + (a >> 2));
        }
    };
    struct frame {
        closure m_closure;
        expr m_expr;
        std::vector<closure> m_dependencies;
        std::size_t m_next = 0;
    };
    reduction_region const & m_region;
    reduction::binding const * m_bindings;
    index m_count;
    std::unordered_map<key, expr, key_hash> m_cache;

    static key identity(closure c) { return {c.m_code, c.m_env}; }

    frame prepare(closure c) const {
        if (c.m_code >= m_region.size() ||
            (c.m_env != reduction::no_binding && c.m_env >= m_count))
            throw std::invalid_argument("invalid reduction closure");
        expr e = m_region.get(c.m_code).m_expr;
        frame f{c, e, {}, 0};
        auto needed = get_loose_bvar_range(e);
        index env = c.m_env;
        // Only bindings which can occur in this syntax are reconstructed.
        // Merely capturing an environment does not demand all of its values.
        while (env != reduction::no_binding && f.m_dependencies.size() < needed) {
            if (env >= m_count) throw std::invalid_argument("invalid reduction environment");
            auto const & b = m_bindings[env];
            if ((b.m_parent != reduction::no_binding && b.m_parent >= env) ||
                (b.m_value.m_env != reduction::no_binding && b.m_value.m_env >= env))
                throw std::invalid_argument("cyclic reduction environment");
            f.m_dependencies.push_back(b.m_value);
            env = b.m_parent;
        }
        return f;
    }

public:
    reduction_reifier(reduction_region const & region,
                      reduction::binding const * bindings, index count):
        m_region(region), m_bindings(bindings), m_count(count) {
        if (count != 0 && bindings == nullptr)
            throw std::invalid_argument("missing reduction bindings");
    }

    expr operator()(closure root) {
        auto cached = m_cache.find(identity(root));
        if (cached != m_cache.end()) return cached->second;
        std::vector<frame> stack;
        stack.push_back(prepare(root));
        while (!stack.empty()) {
            frame & f = stack.back();
            if (f.m_next < f.m_dependencies.size()) {
                closure child = f.m_dependencies[f.m_next++];
                if (m_cache.find(identity(child)) == m_cache.end())
                    stack.push_back(prepare(child));
                continue;
            }
            std::vector<expr> substitutions;
            substitutions.reserve(f.m_dependencies.size());
            for (auto c : f.m_dependencies) substitutions.push_back(m_cache.at(identity(c)));
            // prepare bounds the substitution count by the unsigned range
            // reported by Lean's own expression metadata.
            expr result = instantiate(f.m_expr, static_cast<unsigned>(substitutions.size()),
                                      substitutions.data());
            m_cache.emplace(identity(f.m_closure), result);
            stack.pop_back();
        }
        return m_cache.at(identity(root));
    }
};

/** Reconstruct a completed result or a suspended application spine. Pending
    arguments are applied from the top of the explicit stack, without forcing
    them. The caller remains responsible for the machine outcome/trust policy. */
inline expr reify_reduction_machine(reduction_region const & region,
                                   reduction::machine const & m,
                                   reduction::closure const * arguments,
                                   reduction::index argument_capacity,
                                   reduction::binding const * bindings,
                                   reduction::index binding_capacity) {
    if (m.m_num_arguments > argument_capacity || m.m_num_bindings > binding_capacity ||
        (m.m_num_arguments != 0 && arguments == nullptr))
        throw std::invalid_argument("invalid reduction workspace");
    reduction_reifier quote(region, bindings, m.m_num_bindings);
    expr result = quote(m.m_control);
    for (auto i = m.m_num_arguments; i != 0; --i)
        result = mk_app(result, quote(arguments[i - 1]));
    return result;
}
}
