/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <cstddef>
#include <cstdint>

namespace lean {
namespace reduction {

using index = std::uint64_t;
constexpr index no_binding = UINT64_MAX;

/** Pointer-free instructions for the lazy closure fragment. Values carry
    a source payload ID; this fragment does not interpret numeric payloads.
    Unsupported operations are boundaries, never successful reductions. */
enum class opcode : std::uint8_t { value, bound, app, lambda, let, alias, unsupported };
struct instruction {
    opcode m_op;
    index m_first;
    index m_second;
    index m_source;
};
struct closure { index m_code; index m_env; };
struct binding { closure m_value; index m_parent; };

enum class outcome : std::uint8_t {
    running, complete, unsupported, need_arguments, need_bindings, invalid
};

/** All machine state is explicit, index-based and relocatable. The caller
    owns the instruction array and workspace. Growing/replacing workspace
    while preserving its used prefix allows an exact resource continuation.
    There is no algorithmic step/depth limit. A scheduler may call step any
    number of times without turning a scheduling slice into a proof verdict.

    Environment indices only refer backwards in the binding arena. This
    invariant is checked during lookup, so corrupt workspace cannot induce
    an infinite environment walk. Supported source terms may still diverge;
    semantic termination remains the reference evaluator's responsibility. */
struct machine {
    closure m_control;
    index m_num_arguments = 0;
    index m_num_bindings = 0;

    explicit machine(index root):m_control{root, no_binding} {}

    outcome step(instruction const * code, index code_size,
                 closure * arguments, index argument_capacity,
                 binding * bindings, index binding_capacity) {
        if (m_control.m_code >= code_size || m_num_arguments > argument_capacity ||
            m_num_bindings > binding_capacity ||
            (m_control.m_env != no_binding && m_control.m_env >= m_num_bindings))
            return outcome::invalid;
        instruction const & n = code[m_control.m_code];
        switch (n.m_op) {
        case opcode::unsupported:
            return outcome::unsupported;
        case opcode::value:
            return m_num_arguments == 0 ? outcome::complete : outcome::unsupported;
        case opcode::bound: {
            index env = m_control.m_env;
            index remaining = n.m_first;
            while (env != no_binding) {
                if (env >= m_num_bindings) return outcome::invalid;
                binding const & b = bindings[env];
                if (b.m_parent != no_binding && b.m_parent >= env) return outcome::invalid;
                if (b.m_value.m_env != no_binding && b.m_value.m_env >= env)
                    return outcome::invalid;
                if (remaining == 0) {
                    if (b.m_value.m_code >= code_size) return outcome::invalid;
                    m_control = b.m_value;
                    return outcome::running;
                }
                --remaining;
                env = b.m_parent;
            }
            return outcome::unsupported;
        }
        case opcode::app:
            if (n.m_first >= code_size || n.m_second >= code_size) return outcome::invalid;
            if (m_num_arguments == argument_capacity) return outcome::need_arguments;
            arguments[m_num_arguments++] = {n.m_second, m_control.m_env};
            m_control.m_code = n.m_first;
            return outcome::running;
        case opcode::lambda:
            if (n.m_first >= code_size) return outcome::invalid;
            if (m_num_arguments == 0) return outcome::complete;
            if (m_num_bindings == binding_capacity) return outcome::need_bindings;
            if (arguments[m_num_arguments - 1].m_code >= code_size ||
                (arguments[m_num_arguments - 1].m_env != no_binding &&
                 arguments[m_num_arguments - 1].m_env >= m_num_bindings))
                return outcome::invalid;
            bindings[m_num_bindings] = {arguments[--m_num_arguments], m_control.m_env};
            m_control = {n.m_first, m_num_bindings++};
            return outcome::running;
        case opcode::let:
            if (n.m_first >= code_size || n.m_second >= code_size) return outcome::invalid;
            if (m_num_bindings == binding_capacity) return outcome::need_bindings;
            bindings[m_num_bindings] = {{n.m_first, m_control.m_env}, m_control.m_env};
            m_control = {n.m_second, m_num_bindings++};
            return outcome::running;
        case opcode::alias:
            if (n.m_first >= code_size) return outcome::invalid;
            m_control.m_code = n.m_first;
            return outcome::running;
        }
        return outcome::invalid;
    }
};

}
}
