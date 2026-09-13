/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <cstddef>
#include <cstdint>
#include "kernel/reduction_natural.h"

namespace lean {
namespace reduction {

using index = std::uint64_t;
constexpr index no_binding = UINT64_MAX;

/** Pointer-free instructions for lazy closures and supported exact arithmetic.
    Opaque values retain source IDs; numeric literals refer to immutable limbs.
    Unsupported operations are boundaries, never successful reductions. */
enum class opcode : std::uint8_t {
    value, bound, app, lambda, let, alias, unsupported,
    natural, natural_zero, natural_add, natural_subtract, natural_multiply, definition
};
struct instruction {
    opcode m_op;
    index m_first;
    index m_second;
    index m_source;
};
struct closure { index m_code; index m_env; };
struct binding { closure m_value; index m_parent; };

struct natural_reference {
    index m_offset = 0;
    index m_size = 0;
    bool m_generated = false;
};
struct arithmetic_frame {
    closure m_operator;
    closure m_right;
    opcode m_operation;
    natural_reference m_value;
    bool m_have_left = false;
};
struct arithmetic_memory {
    natural_limb const * m_literals = nullptr;
    index m_literal_count = 0;
    natural_limb * m_values = nullptr;
    index m_value_capacity = 0;
    arithmetic_frame * m_frames = nullptr;
    index m_frame_capacity = 0;
    index m_column_capacity = 0;
};

enum class outcome : std::uint8_t {
    running, complete, unsupported, need_arguments, need_bindings, invalid,
    need_frames, need_values, need_columns, product_ready
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
    index m_num_frames = 0;
    index m_num_values = 0;
    index m_required_values = 0;
    index m_required_columns = 0;
    natural_reference m_value;
    bool m_has_value = false;

    explicit machine(index root):m_control{root, no_binding} {}

    bool valid_reference(natural_reference r, natural_limb const * literals, index literal_count,
                         natural_limb const * values) const {
        auto count = r.m_generated ? m_num_values : literal_count;
        auto data = r.m_generated ? values : literals;
        return r.m_offset <= count && r.m_size <= count - r.m_offset &&
            valid_natural({data == nullptr ? nullptr : data + r.m_offset, r.m_size});
    }

    bool valid_reference(natural_reference r, arithmetic_memory memory) const {
        return valid_reference(r, memory.m_literals, memory.m_literal_count, memory.m_values);
    }

    natural_view view(natural_reference r, arithmetic_memory memory) const {
        auto data = r.m_generated ? memory.m_values : memory.m_literals;
        return {data == nullptr ? nullptr : data + r.m_offset, r.m_size};
    }

    outcome accept_natural_result(natural_result result) {
        if (result.m_status == natural_status::need_space) {
            if (result.m_size > UINT64_MAX - m_num_values) return outcome::invalid;
            m_required_values = m_num_values + result.m_size;
            return outcome::need_values;
        }
        if (result.m_status != natural_status::complete) return outcome::invalid;
        m_value = {m_num_values, result.m_size, true};
        m_num_values += result.m_size;
        m_required_values = 0;
        m_required_columns = 0;
        --m_num_frames;
        return outcome::running;
    }

    /** The backend may distribute a ready product's columns without leaving
        this region. This is a work request, not a completed reduction. Both
        operands and the pending frame remain unchanged until completion. */
    outcome finish_product(arithmetic_memory memory, natural_column const * columns, index count) {
        if (!m_has_value || m_num_arguments != 0 || m_num_frames == 0 ||
            m_num_frames > memory.m_frame_capacity || memory.m_frames == nullptr ||
            m_num_values > memory.m_value_capacity ||
            !valid_reference(m_value, memory)) return outcome::invalid;
        auto const & frame = memory.m_frames[m_num_frames - 1];
        if (!frame.m_have_left || frame.m_operation != opcode::natural_multiply ||
            !valid_reference(frame.m_value, memory)) return outcome::invalid;
        auto a = view(frame.m_value, memory), b = view(m_value, memory);
        if (a.m_size > UINT64_MAX - b.m_size) return outcome::invalid;
        auto expected = a.m_size == 0 || b.m_size == 0 ? 0 : a.m_size + b.m_size;
        if (count != expected || count > memory.m_column_capacity) return outcome::invalid;
        auto destination = memory.m_values == nullptr ? nullptr : memory.m_values + m_num_values;
        return accept_natural_result(finish_natural_product(columns, count, destination,
                                                            memory.m_value_capacity - m_num_values));
    }

    outcome return_value(arithmetic_memory memory, bool defer_products) {
        if (m_num_arguments != 0) return outcome::unsupported;
        if (!valid_reference(m_value, memory)) return outcome::invalid;
        if (m_num_frames == 0) return outcome::complete;
        auto & frame = memory.m_frames[m_num_frames - 1];
        if (!frame.m_have_left) {
            frame.m_value = m_value;
            frame.m_have_left = true;
            m_control = frame.m_right;
            m_has_value = false;
            return outcome::running;
        }
        if (!valid_reference(frame.m_value, memory)) return outcome::invalid;
        auto a = view(frame.m_value, memory), b = view(m_value, memory);
        auto destination = memory.m_values == nullptr ? nullptr : memory.m_values + m_num_values;
        auto available = memory.m_value_capacity - m_num_values;
        if (defer_products && frame.m_operation == opcode::natural_multiply &&
            a.m_size != 0 && b.m_size != 0) {
            if (a.m_size > UINT64_MAX - b.m_size) return outcome::invalid;
            auto count = a.m_size + b.m_size;
            if (available < count) return accept_natural_result({natural_status::need_space, count});
            m_required_values = 0;
            if (memory.m_column_capacity < count) {
                m_required_columns = count;
                return outcome::need_columns;
            }
            m_required_columns = 0;
            return outcome::product_ready;
        }
        natural_result result{natural_status::invalid, 0};
        switch (frame.m_operation) {
        case opcode::natural_add: result = add_natural(a, b, destination, available); break;
        case opcode::natural_subtract: result = subtract_natural(a, b, destination, available); break;
        case opcode::natural_multiply: result = multiply_natural(a, b, destination, available); break;
        default: return outcome::invalid;
        }
        return accept_natural_result(result);
    }

    outcome step(instruction const * code, index code_size,
                 closure * arguments, index argument_capacity,
                 binding * bindings, index binding_capacity,
                 arithmetic_memory memory = {}, bool defer_products = false) {
        if (m_control.m_code >= code_size || m_num_arguments > argument_capacity ||
            m_num_bindings > binding_capacity ||
            m_num_frames > memory.m_frame_capacity || m_num_values > memory.m_value_capacity ||
            (argument_capacity != 0 && arguments == nullptr) ||
            (binding_capacity != 0 && bindings == nullptr) ||
            (memory.m_frame_capacity != 0 && memory.m_frames == nullptr) ||
            (memory.m_value_capacity != 0 && memory.m_values == nullptr) ||
            (memory.m_literal_count != 0 && memory.m_literals == nullptr) ||
            (m_control.m_env != no_binding && m_control.m_env >= m_num_bindings))
            return outcome::invalid;
        if (m_has_value) return return_value(memory, defer_products);
        instruction const & n = code[m_control.m_code];
        switch (n.m_op) {
        case opcode::unsupported:
            return outcome::unsupported;
        case opcode::value:
            return m_num_arguments == 0 && m_num_frames == 0 ? outcome::complete : outcome::unsupported;
        case opcode::natural_zero:
            // The constructor is a numeric operand, but bare Nat.zero must
            // keep its constructor syntax in the reference normal form.
            if (m_num_arguments != 0) return outcome::unsupported;
            if (m_num_frames == 0) return outcome::complete;
            m_value = {};
            m_has_value = true;
            return outcome::running;
        case opcode::natural:
            if (m_num_arguments != 0) return outcome::unsupported;
            m_value = {n.m_first, n.m_second, false};
            m_has_value = true;
            return outcome::running;
        case opcode::natural_add: case opcode::natural_subtract: case opcode::natural_multiply:
            // Exactly the fully applied primitive arity, as in reduce_nat.
            // Strict operands get their own empty application spine. Parent
            // primitives are retained in frames, not recursively run on CPU.
            if (m_num_arguments != 2) return outcome::unsupported;
            if (m_num_frames == memory.m_frame_capacity) return outcome::need_frames;
            memory.m_frames[m_num_frames++] = {
                m_control, arguments[0], n.m_op, {}, false};
            m_control = arguments[1];
            m_num_arguments = 0;
            return outcome::running;
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
            if (m_num_arguments == 0)
                return m_num_frames == 0 ? outcome::complete : outcome::unsupported;
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
        case opcode::definition:
            if (n.m_first >= code_size) return outcome::invalid;
            // Global values capture no caller lexical environment. Arguments
            // retain their own closures, including the caller's bindings.
            m_control = {n.m_first, no_binding};
            return outcome::running;
        }
        return outcome::invalid;
    }
};

}
}
