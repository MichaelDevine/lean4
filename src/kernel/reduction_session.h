/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include "kernel/reduction_reify.h"

namespace lean {

/** Single-owner host checkpoint for the supported closure fragment. Backend
    submissions operate on a private copy and commit only after successful
    synchronization. Exceptions and invalid outcomes leave this checkpoint
    unchanged. This is execution plumbing, not independent result validation.

    Workspace sizes are supplied by resource admission, not an algorithmic
    limit. A resource outcome can be resumed after resize_workspace. There is
    no automatic allocation policy, device selection or proof verdict here. */
class reduction_session {
    reduction_region m_region;
    std::vector<reduction::instruction> m_code;
    reduction::machine m_machine;
    std::vector<reduction::closure> m_arguments;
    std::vector<reduction::binding> m_bindings;
public:
    reduction_session(expr const & root, std::size_t arguments, std::size_t bindings):
        m_region(root), m_code(make_reduction_program(m_region)),
        m_machine(m_region.root()), m_arguments(arguments), m_bindings(bindings) {}

    reduction::machine const & state() const { return m_machine; }
    std::size_t argument_capacity() const { return m_arguments.size(); }
    std::size_t binding_capacity() const { return m_bindings.size(); }
    std::size_t program_size() const { return m_code.size(); }

    void resize_workspace(std::size_t arguments, std::size_t bindings) {
        if (arguments < m_machine.m_num_arguments || bindings < m_machine.m_num_bindings)
            throw std::invalid_argument("cannot discard live reduction workspace");
        // Both allocations must succeed before either checkpoint array changes.
        auto new_arguments = m_arguments;
        auto new_bindings = m_bindings;
        new_arguments.resize(arguments);
        new_bindings.resize(bindings);
        m_arguments.swap(new_arguments);
        m_bindings.swap(new_bindings);
    }

    template<typename Backend> reduction::outcome advance(Backend & backend) {
        auto next = m_machine;
        auto arguments = m_arguments;
        auto bindings = m_bindings;
        auto status = backend(m_code, next, arguments, bindings);
        using reduction::outcome;
        switch (status) {
        case outcome::running: case outcome::complete: case outcome::unsupported:
        case outcome::need_arguments: case outcome::need_bindings:
            break;
        default:
            return outcome::invalid;
        }
        if (next.m_control.m_code >= m_code.size() ||
            next.m_num_arguments > arguments.size() || next.m_num_bindings > bindings.size() ||
            (next.m_control.m_env != reduction::no_binding &&
             next.m_control.m_env >= next.m_num_bindings))
            return outcome::invalid;
        m_arguments.swap(arguments);
        m_bindings.swap(bindings);
        m_machine = next;
        return status;
    }

    expr reconstruct() const {
        return reify_reduction_machine(m_region, m_machine, m_arguments.data(),
            m_arguments.size(), m_bindings.data(), m_bindings.size());
    }
};

/** CPU implementation of the same backend boundary. This is not the full
    reference type checker; unsupported reductions remain explicit. */
struct reduction_cpu_backend {
    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings) const {
        reduction::outcome result;
        do {
            result = state.step(code.data(), code.size(), arguments.data(), arguments.size(),
                                bindings.data(), bindings.size());
        } while (result == reduction::outcome::running);
        return result;
    }
};
}
