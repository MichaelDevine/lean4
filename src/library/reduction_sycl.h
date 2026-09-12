/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <memory>
#include <vector>
#include "kernel/reduction_machine.h"

namespace lean {

/** Optional GPU backend, compiled separately from Lean's host runtime.
    Neither SYCL nor Lean object types cross this boundary. Device selection
    uses SYCL's GPU selector (and runtime configuration), never a CPU fallback.
    Exceptions propagate to the session's transactional submission boundary.

    This initial transport uploads each submission. Persistent device storage,
    batching and integration into the type checker remain unfinished work. */
class reduction_sycl_backend {
    class imp;
    std::unique_ptr<imp> m_imp;
public:
    reduction_sycl_backend();
    ~reduction_sycl_backend();
    reduction_sycl_backend(reduction_sycl_backend const &) = delete;
    reduction_sycl_backend & operator=(reduction_sycl_backend const &) = delete;
    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings);
};
}
