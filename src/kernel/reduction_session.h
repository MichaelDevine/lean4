/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <unordered_set>
#include "kernel/reduction_reify.h"
#include "kernel/reduction_storage.h"

namespace lean {

/** Single-owner host checkpoint for supported closures and exact arithmetic. Backend
    submissions operate on a private copy and commit only after successful
    synchronization. Exceptions and invalid outcomes leave this checkpoint
    unchanged. This is execution plumbing, not independent result validation.

    Workspace sizes are supplied by resource admission, not an algorithmic
    limit. A resource outcome can be resumed after resize_workspace. There is
    no automatic allocation policy, device selection or proof verdict here.
    The optional environment is read only during lowering; it is not retained
    as a borrowed pointer or sent to the backend. Primitive arithmetic requires
    that semantic snapshot. The result arena currently grows monotonically. */
class reduction_session {
    reduction_region m_region;
    std::vector<reduction::natural_limb> m_literals;
    std::vector<reduction::instruction> m_code;
    reduction::machine m_machine;
    std::vector<reduction::closure> m_arguments;
    std::vector<reduction::binding> m_bindings;
    std::vector<reduction::natural_limb> m_values;
    std::vector<reduction::arithmetic_frame> m_frames;
    std::size_t m_columns = 0;

    struct candidate {
        reduction::machine m_state;
        std::vector<reduction::closure> m_arguments;
        std::vector<reduction::binding> m_bindings;
        std::vector<reduction::natural_limb> m_values;
        std::vector<reduction::arithmetic_frame> m_frames;

        explicit candidate(reduction_session const & s):m_state(s.m_machine),
            m_arguments(s.m_arguments), m_bindings(s.m_bindings),
            m_values(s.m_values), m_frames(s.m_frames) {}

        reduction::submission request(reduction_session const & s) {
            return {s.m_code, m_state, m_arguments, m_bindings,
                    {s.m_literals, m_values, m_frames, s.m_columns}};
        }
    };

    reduction::outcome commit(candidate & c, reduction::outcome status) {
        using reduction::outcome;
        switch (status) {
        case outcome::running: case outcome::complete: case outcome::unsupported:
        case outcome::need_arguments: case outcome::need_bindings:
        case outcome::need_frames: case outcome::need_values: case outcome::need_columns:
            break;
        default:
            return outcome::invalid;
        }
        auto const & next = c.m_state;
        auto arithmetic = c.request(*this).m_arithmetic;
        if (next.m_control.m_code >= m_code.size() ||
            next.m_num_arguments > c.m_arguments.size() || next.m_num_bindings > c.m_bindings.size() ||
            next.m_num_frames > c.m_frames.size() || next.m_num_values > c.m_values.size() ||
            (next.m_has_value && !next.valid_reference(next.m_value, arithmetic.memory())) ||
            (status == outcome::complete && (next.m_num_frames != 0 || next.m_num_arguments != 0)) ||
            (next.m_control.m_env != reduction::no_binding &&
             next.m_control.m_env >= next.m_num_bindings))
            return outcome::invalid;
        m_arguments.swap(c.m_arguments);
        m_bindings.swap(c.m_bindings);
        m_frames.swap(c.m_frames);
        m_values.swap(c.m_values);
        m_machine = next;
        return status;
    }
public:
    reduction_session(expr const & root, std::size_t arguments, std::size_t bindings,
                      reduction_mode mode = reduction_mode::full(), environment const * env = nullptr):
        m_region(root), m_code(make_reduction_program(m_region, m_literals, mode, env)),
        m_machine(m_region.root()), m_arguments(arguments), m_bindings(bindings) {}

    reduction::machine const & state() const { return m_machine; }
    std::size_t argument_capacity() const { return m_arguments.size(); }
    std::size_t binding_capacity() const { return m_bindings.size(); }
    std::size_t program_size() const { return m_code.size(); }
    std::size_t frame_capacity() const { return m_frames.size(); }
    std::size_t value_capacity() const { return m_values.size(); }
    std::size_t literal_size() const { return m_literals.size(); }
    std::size_t column_capacity() const { return m_columns; }

    void resize_workspace(std::size_t arguments, std::size_t bindings,
                          std::size_t frames = 0, std::size_t values = 0, std::size_t columns = 0) {
        if (arguments < m_machine.m_num_arguments || bindings < m_machine.m_num_bindings ||
            frames < m_machine.m_num_frames || values < m_machine.m_num_values)
            throw std::invalid_argument("cannot discard live reduction workspace");
        // All allocations must succeed before any checkpoint array changes.
        auto new_arguments = m_arguments;
        auto new_bindings = m_bindings;
        auto new_frames = m_frames;
        auto new_values = m_values;
        new_arguments.resize(arguments);
        new_bindings.resize(bindings);
        new_frames.resize(frames);
        new_values.resize(values);
        m_arguments.swap(new_arguments);
        m_bindings.swap(new_bindings);
        m_frames.swap(new_frames);
        m_values.swap(new_values);
        m_columns = columns;
    }

    template<typename Backend> reduction::outcome advance(Backend & backend) {
        candidate c(*this);
        auto r = c.request(*this);
        return commit(c, backend(r.m_code, r.m_state, r.m_arguments, r.m_bindings, r.m_arithmetic));
    }

    /** Execute independent captured regions together, preserving input order.
        No cross-region mutable references or duplicate sessions are allowed.
        An exception or malformed outcome count commits nothing. Individual
        invalid outcomes leave that session unchanged; valid siblings can
        still finish or save a resource continuation. There is no batch-size
        ceiling or implicit scheduling policy here. */
    template<typename Backend>
    static std::vector<reduction::outcome> advance_batch(
        std::vector<reduction_session *> const & sessions, Backend & backend) {
        if (sessions.empty()) return {};
        std::unordered_set<reduction_session *> distinct;
        for (auto s : sessions)
            if (!s || !distinct.insert(s).second)
                throw std::invalid_argument("batch sessions must be nonnull and distinct");
        std::vector<candidate> candidates;
        candidates.reserve(sessions.size());
        for (auto s : sessions) candidates.emplace_back(*s);
        std::vector<reduction::submission> requests;
        requests.reserve(sessions.size());
        for (std::size_t i = 0; i < sessions.size(); ++i)
            requests.push_back(candidates[i].request(*sessions[i]));
        auto results = backend.submit(requests);
        if (results.size() != sessions.size())
            return std::vector<reduction::outcome>(sessions.size(), reduction::outcome::invalid);
        for (std::size_t i = 0; i < sessions.size(); ++i)
            results[i] = sessions[i]->commit(candidates[i], results[i]);
        return results;
    }

    expr reconstruct() const {
        auto number = [&](reduction::natural_reference r) {
            if (!m_machine.valid_reference(r, m_literals.data(), m_literals.size(), m_values.data()))
                throw std::invalid_argument("invalid reconstructed numeric reference");
            auto const & arena = r.m_generated ? m_values : m_literals;
            return reify_reduction_natural({arena.empty() ? nullptr : arena.data() + r.m_offset, r.m_size});
        };
        expr result = m_machine.m_has_value ? number(m_machine.m_value) :
            reify_reduction_machine(m_region, m_machine, m_arguments.data(),
                m_arguments.size(), m_bindings.data(), m_bindings.size());
        reduction_reifier quote(m_region, m_bindings.data(), m_machine.m_num_bindings);
        for (auto i = m_machine.m_num_frames; i != 0; --i) {
            auto const & frame = m_frames.at(i - 1);
            if (frame.m_have_left)
                result = mk_app(mk_app(quote(frame.m_operator), number(frame.m_value)), result);
            else
                result = mk_app(mk_app(quote(frame.m_operator), result), quote(frame.m_right));
        }
        return result;
    }
};

/** CPU implementation of the same backend boundary. This is not the full
    reference type checker; unsupported reductions remain explicit. */
struct reduction_cpu_backend {
    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings,
                                  reduction::arithmetic_workspace & arithmetic) const {
        reduction::outcome result;
        do {
            result = state.step(code.data(), code.size(), arguments.data(), arguments.size(),
                                bindings.data(), bindings.size(), arithmetic.memory());
        } while (result == reduction::outcome::running);
        return result;
    }

    // Deterministic serial reference control, not a production CPU scheduler.
    std::vector<reduction::outcome> submit(std::vector<reduction::submission> & requests) const {
        std::vector<reduction::outcome> results;
        results.reserve(requests.size());
        for (auto & r : requests)
            results.push_back((*this)(r.m_code, r.m_state, r.m_arguments, r.m_bindings, r.m_arithmetic));
        return results;
    }
};
}
