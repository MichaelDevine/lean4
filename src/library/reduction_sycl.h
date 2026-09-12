/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <memory>
#include <vector>
#include "kernel/reduction_storage.h"

namespace lean {

/** Optional GPU backend, compiled separately from Lean's host runtime.
    Neither SYCL nor Lean object types cross this boundary. Device selection
    uses SYCL's GPU selector (and runtime configuration), never a CPU fallback.
    Device creation is lazy; platform exceptions become reduction_backend_error
    at the session's transactional submission boundary.

    This initial transport uploads each submission and evaluates one region
    with one work-item. Exact nested arithmetic is integrated; cooperative
    arithmetic, persistent device storage and batched regions remain open. */
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
                                  std::vector<reduction::binding> & bindings,
                                  reduction::arithmetic_workspace & arithmetic);
};
}
