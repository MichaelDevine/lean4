/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <memory>
#include <optional>
#include <vector>
#include "kernel/reduction_storage.h"

namespace lean {

enum class reduction_sycl_execution { scalar, cooperative };

/** Optional GPU backend, compiled separately from Lean's host runtime.
    Neither SYCL nor Lean object types cross this boundary. Device selection
    uses SYCL's GPU selector (and runtime configuration), never a CPU fallback.
    Device creation is lazy; platform exceptions become reduction_backend_error
    at the session's transactional submission boundary.

    Cooperative execution keeps control on a group leader and distributes
    product columns across the group. Scalar execution is retained as an
    explicit reference/benchmark option, not a silent device fallback.
    A batch assigns one independent region to each group. Every region keeps
    its own status and workspace; no device-wide barrier couples progress.
    Submission is currently synchronous: results become host-visible together
    after all groups stop. This is not asynchronous checkpoint publication.
    This transport still uploads each submission; residency is unfinished. */
class reduction_sycl_backend {
    class imp;
    std::unique_ptr<imp> m_imp;
    reduction_sycl_execution m_execution;
    bool m_profiling;
public:
    explicit reduction_sycl_backend(reduction_sycl_execution execution = reduction_sycl_execution::cooperative,
                                    bool profiling = false);
    ~reduction_sycl_backend();
    reduction_sycl_backend(reduction_sycl_backend const &) = delete;
    reduction_sycl_backend & operator=(reduction_sycl_backend const &) = delete;
    std::size_t work_group_size() const;
    std::uint64_t last_cooperative_products() const;
    std::size_t last_region_count() const;
    // Optional device-event measurement. Disabled by default; never a result
    // validity condition or an estimate of complete checker cost.
    std::optional<std::uint64_t> last_kernel_nanoseconds() const;
    std::vector<reduction::outcome> submit(std::vector<reduction::submission> & requests);
    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings,
                                  reduction::arithmetic_workspace & arithmetic);
};
}
