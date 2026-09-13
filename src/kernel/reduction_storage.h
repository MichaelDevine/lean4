/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <vector>
#include "kernel/reduction_machine.h"

namespace lean {
namespace reduction {

/** Borrowed host submission workspace. Literal inputs are immutable;
    generated values and strict-operand frames belong to the session's
    transactional candidate. No host vector or pointer enters device state. */
struct arithmetic_workspace {
    std::vector<natural_limb> const & m_literals;
    std::vector<natural_limb> & m_values;
    std::vector<arithmetic_frame> & m_frames;
    // Scratch is not checkpoint data: it is rebuilt after a resource stop.
    std::size_t m_columns = 0;

    arithmetic_memory memory() const {
        return {m_literals.data(), m_literals.size(), m_values.data(), m_values.size(),
                m_frames.data(), m_frames.size(), m_columns};
    }
};

}
}
