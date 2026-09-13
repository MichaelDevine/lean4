/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#include <algorithm>
#include <sycl/sycl.hpp>
#include "library/reduction_sycl.h"
#include "kernel/reduction_backend_error.h"

namespace lean {

class reduction_scalar_kernel;
class reduction_cooperative_kernel;

namespace {
struct buffer_slice { std::size_t m_offset, m_size; };
struct region_layout {
    buffer_slice m_code, m_arguments, m_bindings, m_literals, m_values, m_frames, m_columns;
};

template<typename T>
buffer_slice append_region_buffer(std::vector<T> & buffer, std::vector<T> const & region) {
    buffer_slice result{buffer.size(), region.size()};
    buffer.insert(buffer.end(), region.begin(), region.end());
    return result;
}

template<typename T> void ensure_device_storage(std::vector<T> & buffer) {
    // Sentinels satisfy SYCL allocation requirements, not logical capacity.
    if (buffer.empty()) buffer.resize(1);
}

template<typename T>
void restore_region_buffer(std::vector<T> const & buffer, buffer_slice slice, std::vector<T> & region) {
    std::copy_n(buffer.begin() + slice.m_offset, slice.m_size, region.begin());
}
}

class reduction_sycl_backend::imp {
    sycl::queue m_queue;
    reduction_sycl_execution m_execution;
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> m_bundle;
    std::size_t m_group_size = 0;
    std::uint64_t m_last_products = 0;
    std::size_t m_last_regions = 0;
    bool m_profiling;
    std::optional<std::uint64_t> m_last_kernel_ns;

    void prepare_group() {
        if (m_bundle) return;
        auto device = m_queue.get_device();
        auto id = sycl::get_kernel_id<reduction_cooperative_kernel>();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            m_queue.get_context(), {device}, {id});
        auto maximum = bundle.get_kernel(id).get_info<sycl::info::kernel_device_specific::work_group_size>(device);
        auto dimension = device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
        auto size = std::min(maximum, dimension);
        if (size == 0) throw reduction_backend_error("device reported an empty work-group capacity");
        m_bundle.reset(new sycl::kernel_bundle<sycl::bundle_state::executable>(std::move(bundle)));
        m_group_size = size;
    }
public:
    explicit imp(reduction_sycl_execution execution, bool profiling):
        m_queue(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto const & error : errors) std::rethrow_exception(error);
        }, profiling ? sycl::property_list{sycl::property::queue::enable_profiling{}} : sycl::property_list{}),
        m_execution(execution), m_profiling(profiling) {}

    std::size_t work_group_size() const {
        return m_execution == reduction_sycl_execution::scalar ? 1 : m_group_size;
    }
    std::uint64_t last_cooperative_products() const { return m_last_products; }
    std::size_t last_region_count() const { return m_last_regions; }
    std::optional<std::uint64_t> last_kernel_nanoseconds() const { return m_last_kernel_ns; }

    std::vector<reduction::outcome> submit(std::vector<reduction::submission> & requests) {
        using namespace reduction;
        m_last_products = 0;
        m_last_regions = 0;
        m_last_kernel_ns.reset();
        if (requests.empty()) return {};
        bool cooperative = m_execution == reduction_sycl_execution::cooperative;
        if (cooperative) prepare_group();
        auto group_size = work_group_size();
        if (requests.size() > SIZE_MAX / group_size)
            throw std::length_error("reduction batch work range overflow");
        std::vector<instruction> device_code;
        std::vector<closure> device_arguments;
        std::vector<binding> device_bindings;
        std::vector<natural_limb> device_literals, device_values;
        std::vector<arithmetic_frame> device_frames;
        std::vector<machine> next;
        std::vector<region_layout> layout;
        std::vector<outcome> results(requests.size(), outcome::invalid);
        std::vector<std::uint64_t> products(requests.size(), 0);
        std::size_t column_capacity = 0;
        std::optional<std::uint64_t> kernel_ns;
        next.reserve(requests.size()); layout.reserve(requests.size());
        for (auto const & r : requests) {
            auto columns = cooperative ? r.m_arithmetic.m_columns : 0;
            if (columns > SIZE_MAX / sizeof(natural_column) - column_capacity)
                throw std::length_error("reduction batch scratch overflow");
            layout.push_back({
                append_region_buffer(device_code, r.m_code),
                append_region_buffer(device_arguments, r.m_arguments),
                append_region_buffer(device_bindings, r.m_bindings),
                append_region_buffer(device_literals, r.m_arithmetic.m_literals),
                append_region_buffer(device_values, r.m_arithmetic.m_values),
                append_region_buffer(device_frames, r.m_arithmetic.m_frames),
                {column_capacity, columns}});
            column_capacity += columns;
            next.push_back(r.m_state);
        }
        ensure_device_storage(device_code);
        ensure_device_storage(device_arguments);
        ensure_device_storage(device_bindings);
        ensure_device_storage(device_literals);
        ensure_device_storage(device_values);
        ensure_device_storage(device_frames);
        {
            sycl::buffer<region_layout, 1> regions(layout.data(), sycl::range<1>(layout.size()));
            sycl::buffer<instruction, 1> input(device_code.data(), sycl::range<1>(device_code.size()));
            sycl::buffer<closure, 1> args(device_arguments.data(), sycl::range<1>(device_arguments.size()));
            sycl::buffer<binding, 1> env(device_bindings.data(), sycl::range<1>(device_bindings.size()));
            sycl::buffer<natural_limb, 1> literals(device_literals.data(), sycl::range<1>(device_literals.size()));
            sycl::buffer<natural_limb, 1> values(device_values.data(), sycl::range<1>(device_values.size()));
            sycl::buffer<arithmetic_frame, 1> frames(device_frames.data(), sycl::range<1>(device_frames.size()));
            sycl::buffer<machine, 1> control(next.data(), sycl::range<1>(next.size()));
            sycl::buffer<outcome, 1> output(results.data(), sycl::range<1>(results.size()));
            sycl::buffer<std::uint64_t, 1> product_count(products.data(), sycl::range<1>(products.size()));
            std::unique_ptr<sycl::buffer<natural_column, 1>> columns;
            if (cooperative)
                columns.reset(new sycl::buffer<natural_column, 1>(sycl::range<1>(std::max<std::size_t>(1, column_capacity))));
            auto completion = m_queue.submit([&](sycl::handler & h) {
                auto slices = regions.get_access<sycl::access::mode::read>(h);
                auto instructions = input.get_access<sycl::access::mode::read>(h);
                auto a = args.get_access<sycl::access::mode::read_write>(h);
                auto b = env.get_access<sycl::access::mode::read_write>(h);
                auto l = literals.get_access<sycl::access::mode::read>(h);
                auto v = values.get_access<sycl::access::mode::read_write>(h);
                auto f = frames.get_access<sycl::access::mode::read_write>(h);
                auto m = control.get_access<sycl::access::mode::read_write>(h);
                auto out = output.get_access<sycl::access::mode::write>(h);
                auto completed_products = product_count.get_access<sycl::access::mode::write>(h);
                auto memory = [=](std::size_t region) {
                    auto const & s = slices[region];
                    return arithmetic_memory{
                        l.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_literals.m_offset, s.m_literals.m_size,
                        v.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_values.m_offset, s.m_values.m_size,
                        f.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_frames.m_offset, s.m_frames.m_size,
                        s.m_columns.m_size};
                };
                auto step = [=](std::size_t region, bool defer) {
                    auto const & s = slices[region];
                    return m[region].step(
                            instructions.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_code.m_offset, s.m_code.m_size,
                            a.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_arguments.m_offset, s.m_arguments.m_size,
                            b.get_multi_ptr<sycl::access::decorated::no>().get() + s.m_bindings.m_offset, s.m_bindings.m_size,
                            memory(region), defer);
                };
                if (!cooperative) {
                    h.parallel_for<reduction_scalar_kernel>(sycl::range<1>(requests.size()), [=](sycl::id<1> item) {
                        auto region = item[0];
                        outcome status;
                        do { status = step(region, false); } while (status == outcome::running);
                        out[region] = status;
                        completed_products[region] = 0;
                    });
                } else {
                    h.use_kernel_bundle(*m_bundle);
                    auto c = columns->get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for<reduction_cooperative_kernel>(
                        sycl::nd_range<1>(sycl::range<1>(requests.size() * group_size), sycl::range<1>(group_size)),
                        [=](sycl::nd_item<1> item) {
                            auto group = item.get_group();
                            auto lane = item.get_local_linear_id();
                            auto region = item.get_group_linear_id();
                            auto data = memory(region);
                            auto column_offset = slices[region].m_columns.m_offset;
                            outcome status = outcome::running;
                            std::uint64_t count = 0;
                            while (true) {
                                if (lane == 0 && status == outcome::running) {
                                    do { status = step(region, true); } while (status == outcome::running);
                                }
                                status = static_cast<outcome>(sycl::group_broadcast(group, static_cast<unsigned>(status), 0));
                                // Publish leader-written frames/values before any
                                // worker reads them. All lanes reach every barrier.
                                sycl::group_barrier(group);
                                if (status != outcome::product_ready) break;
                                auto const & frame = data.m_frames[m[region].m_num_frames - 1];
                                auto left = m[region].view(frame.m_value, data);
                                auto right = m[region].view(m[region].m_value, data);
                                auto size = left.m_size + right.m_size;
                                for (std::size_t column = lane; column < size;) {
                                    c[column_offset + column] = multiply_natural_column(left, right, column);
                                    if (size - column <= group_size) break;
                                    column += group_size;
                                }
                                // Consume columns only after every writer finishes.
                                sycl::group_barrier(group);
                                if (lane == 0) {
                                    status = m[region].finish_product(data,
                                        c.get_multi_ptr<sycl::access::decorated::no>().get() + column_offset, size);
                                    if (status == outcome::running && count != UINT64_MAX) ++count;
                                }
                                // Nobody retains reads of the old frame or operands
                                // when the leader advances to the next reduction.
                                sycl::group_barrier(group);
                            }
                            if (lane == 0) { out[region] = status; completed_products[region] = count; }
                        });
                }
            });
            m_queue.wait_and_throw();
            if (m_profiling) {
                auto start = completion.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto finish = completion.get_profiling_info<sycl::info::event_profiling::command_end>();
                if (finish < start) throw reduction_backend_error("device returned reversed profiling timestamps");
                kernel_ns = finish - start;
            }
        }
        // Buffer destruction has completed host copy-back. Surface any async
        // errors before exposing the candidate state to the session commit.
        m_queue.throw_asynchronous();
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto & r = requests[i];
            auto const & s = layout[i];
            restore_region_buffer(device_arguments, s.m_arguments, r.m_arguments);
            restore_region_buffer(device_bindings, s.m_bindings, r.m_bindings);
            restore_region_buffer(device_values, s.m_values, r.m_arithmetic.m_values);
            restore_region_buffer(device_frames, s.m_frames, r.m_arithmetic.m_frames);
            r.m_state = next[i];
            m_last_products += std::min(products[i], UINT64_MAX - m_last_products);
        }
        m_last_regions = requests.size();
        m_last_kernel_ns = kernel_ns;
        return results;
    }
};

reduction_sycl_backend::reduction_sycl_backend(reduction_sycl_execution execution, bool profiling):
    m_execution(execution), m_profiling(profiling) {}
reduction_sycl_backend::~reduction_sycl_backend() = default;

std::size_t reduction_sycl_backend::work_group_size() const { return m_imp ? m_imp->work_group_size() : 0; }
std::uint64_t reduction_sycl_backend::last_cooperative_products() const {
    return m_imp ? m_imp->last_cooperative_products() : 0;
}
std::size_t reduction_sycl_backend::last_region_count() const { return m_imp ? m_imp->last_region_count() : 0; }
std::optional<std::uint64_t> reduction_sycl_backend::last_kernel_nanoseconds() const {
    return m_imp ? m_imp->last_kernel_nanoseconds() : std::nullopt;
}

std::vector<reduction::outcome> reduction_sycl_backend::submit(std::vector<reduction::submission> & requests) {
    try {
        // Empty batches need no GPU, including when no platform is available.
        if (!m_imp && requests.empty()) return {};
        if (!m_imp) m_imp.reset(new imp(m_execution, m_profiling));
        return m_imp->submit(requests);
    } catch (sycl::exception const & error) {
        throw reduction_backend_error(error.what());
    }
}

reduction::outcome reduction_sycl_backend::operator()(
    std::vector<reduction::instruction> const & code, reduction::machine & state,
    std::vector<reduction::closure> & arguments, std::vector<reduction::binding> & bindings,
    reduction::arithmetic_workspace & arithmetic) {
    std::vector<reduction::submission> requests{{code, state, arguments, bindings, arithmetic}};
    return submit(requests).front();
}
}
