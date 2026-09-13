/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <new>
#include <string>
#include "kernel/reduction_extension.h"
#include "kernel/reduction_session.h"
#include "kernel/reduction_backend_error.h"

namespace lean {

struct reduction_workspace {
    std::size_t m_arguments;
    std::size_t m_bindings;
    std::size_t m_frames = 0;
    std::size_t m_values = 0;
    std::size_t m_columns = 0;
};

enum class reduction_attempt {
    complete, unsupported, declined, invalid, invalid_admission, device_failure, memory_failure
};

/** Connect the supported closure machine to kernel dispatch. The borrowed
    backend and admission policy must outlive this thread-owned extension.
    Admission returns an optional workspace for the initial request (running)
    and every resource boundary. It can inspect program size, live workspace,
    current device/memory availability and measured profitability. There is no
    fixed workspace ceiling, retry count or implicit algorithmic timeout.

    Only a complete closure result is returned to the kernel. Other outcomes
    leave the original expression to the ordinary reducer, with the reason
    retained here. Backend errors are not proof rejection. Unexpected exception
    types propagate rather than hiding implementation defects. This adapter,
    evaluator and backend are trusted when installed; this is not a validator. */
template<typename Backend, typename Admission>
class closure_reduction_extension : public reduction_extension {
    Backend & m_backend;
    Admission & m_admission;
    reduction_attempt m_last = reduction_attempt::declined;
    std::string m_device_error;

    bool admit(reduction_session & session, reduction::outcome reason) {
        auto workspace = m_admission(session, reason);
        if (!workspace) { m_last = reduction_attempt::declined; return false; }
        auto const & state = session.state();
        if (workspace->m_arguments < state.m_num_arguments ||
            workspace->m_bindings < state.m_num_bindings ||
            workspace->m_frames < state.m_num_frames ||
            workspace->m_values < state.m_num_values ||
            (reason == reduction::outcome::need_frames &&
             workspace->m_frames <= session.frame_capacity()) ||
            (reason == reduction::outcome::need_values &&
             workspace->m_values <= session.value_capacity()) ||
            (reason == reduction::outcome::need_columns &&
             workspace->m_columns <= session.column_capacity()) ||
            (reason == reduction::outcome::need_arguments &&
             workspace->m_arguments <= session.argument_capacity()) ||
            (reason == reduction::outcome::need_bindings &&
             workspace->m_bindings <= session.binding_capacity())) {
            m_last = reduction_attempt::invalid_admission;
            return false;
        }
        session.resize_workspace(workspace->m_arguments, workspace->m_bindings,
                                 workspace->m_frames, workspace->m_values, workspace->m_columns);
        return true;
    }

public:
    closure_reduction_extension(Backend & backend, Admission & admission):
        m_backend(backend), m_admission(admission) {}
    reduction_attempt last_attempt() const { return m_last; }
    std::string const & device_error() const { return m_device_error; }

    optional<expr> reduce(environment const & env, local_ctx const &, expr const & e,
                          reduction_mode mode) override {
        m_last = reduction_attempt::declined;
        m_device_error.clear();
        try {
            // Numeric primitives and captured definition unfolding are
            // full-mode only. Projection/recursor evaluation is not yet
            // supported; unused captured arguments are still not demanded.
            reduction_session session(e, 0, 0, mode, &env);
            if (!admit(session, reduction::outcome::running)) return none_expr();
            while (true) {
                auto status = session.advance(m_backend);
                switch (status) {
                case reduction::outcome::complete: {
                    expr result = session.reconstruct();
                    m_last = reduction_attempt::complete;
                    return some_expr(result);
                }
                case reduction::outcome::running:
                case reduction::outcome::need_arguments:
                case reduction::outcome::need_bindings:
                case reduction::outcome::need_frames:
                case reduction::outcome::need_values:
                case reduction::outcome::need_columns:
                    if (!admit(session, status)) return none_expr();
                    break;
                case reduction::outcome::unsupported:
                    m_last = reduction_attempt::unsupported; return none_expr();
                case reduction::outcome::invalid:
                case reduction::outcome::product_ready:
                    m_last = reduction_attempt::invalid; return none_expr();
                }
            }
        } catch (reduction_backend_error const & error) {
            m_last = reduction_attempt::device_failure;
            m_device_error = error.what();
        } catch (std::bad_alloc const &) {
            m_last = reduction_attempt::memory_failure;
        }
        return none_expr();
    }
};
}
