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

class reduction_sycl_backend::imp {
    sycl::queue m_queue;
    reduction_sycl_execution m_execution;
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> m_bundle;
    std::size_t m_group_size = 0;
    std::uint64_t m_last_products = 0;

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
    explicit imp(reduction_sycl_execution execution):
        m_queue(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto const & error : errors) std::rethrow_exception(error);
        }), m_execution(execution) {}

    std::size_t work_group_size() const {
        return m_execution == reduction_sycl_execution::scalar ? 1 : m_group_size;
    }
    std::uint64_t last_cooperative_products() const { return m_last_products; }

    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings,
                                  reduction::arithmetic_workspace & arithmetic) {
        using namespace reduction;
        m_last_products = 0;
        if (code.empty()) return outcome::invalid;
        bool cooperative = m_execution == reduction_sycl_execution::cooperative;
        if (cooperative) prepare_group();
        auto group_size = work_group_size();
        auto code_size = code.size();
        auto argument_capacity = arguments.size();
        auto binding_capacity = bindings.size();
        auto literal_count = arithmetic.m_literals.size();
        auto value_capacity = arithmetic.m_values.size();
        auto frame_capacity = arithmetic.m_frames.size();
        auto column_capacity = arithmetic.m_columns;
        // SYCL buffers require a nonempty allocation. Sentinel storage does
        // not enlarge the logical capacity passed to the reduction machine.
        std::vector<closure> device_arguments = arguments;
        std::vector<binding> device_bindings = bindings;
        std::vector<natural_limb> device_literals = arithmetic.m_literals;
        std::vector<natural_limb> device_values = arithmetic.m_values;
        std::vector<arithmetic_frame> device_frames = arithmetic.m_frames;
        if (device_arguments.empty()) device_arguments.resize(1);
        if (device_bindings.empty()) device_bindings.resize(1);
        if (device_literals.empty()) device_literals.resize(1);
        if (device_values.empty()) device_values.resize(1);
        if (device_frames.empty()) device_frames.resize(1);
        auto next = state;
        outcome result = outcome::invalid;
        std::uint64_t products = 0;
        {
            sycl::buffer<instruction, 1> input(code.data(), sycl::range<1>(code_size));
            sycl::buffer<closure, 1> args(device_arguments.data(), sycl::range<1>(device_arguments.size()));
            sycl::buffer<binding, 1> env(device_bindings.data(), sycl::range<1>(device_bindings.size()));
            sycl::buffer<natural_limb, 1> literals(device_literals.data(), sycl::range<1>(device_literals.size()));
            sycl::buffer<natural_limb, 1> values(device_values.data(), sycl::range<1>(device_values.size()));
            sycl::buffer<arithmetic_frame, 1> frames(device_frames.data(), sycl::range<1>(device_frames.size()));
            sycl::buffer<machine, 1> control(&next, sycl::range<1>(1));
            sycl::buffer<outcome, 1> output(&result, sycl::range<1>(1));
            sycl::buffer<std::uint64_t, 1> product_count(&products, sycl::range<1>(1));
            std::unique_ptr<sycl::buffer<natural_column, 1>> columns;
            if (cooperative)
                columns.reset(new sycl::buffer<natural_column, 1>(sycl::range<1>(std::max<std::size_t>(1, column_capacity))));
            m_queue.submit([&](sycl::handler & h) {
                auto instructions = input.get_access<sycl::access::mode::read>(h);
                auto a = args.get_access<sycl::access::mode::read_write>(h);
                auto b = env.get_access<sycl::access::mode::read_write>(h);
                auto l = literals.get_access<sycl::access::mode::read>(h);
                auto v = values.get_access<sycl::access::mode::read_write>(h);
                auto f = frames.get_access<sycl::access::mode::read_write>(h);
                auto m = control.get_access<sycl::access::mode::read_write>(h);
                auto out = output.get_access<sycl::access::mode::write>(h);
                auto completed_products = product_count.get_access<sycl::access::mode::write>(h);
                auto memory = [=]() {
                    return arithmetic_memory{
                        l.get_multi_ptr<sycl::access::decorated::no>().get(), literal_count,
                        v.get_multi_ptr<sycl::access::decorated::no>().get(), value_capacity,
                        f.get_multi_ptr<sycl::access::decorated::no>().get(), frame_capacity, column_capacity};
                };
                auto step = [=](bool defer) {
                    return m[0].step(
                            instructions.get_multi_ptr<sycl::access::decorated::no>().get(), code_size,
                            a.get_multi_ptr<sycl::access::decorated::no>().get(), argument_capacity,
                            b.get_multi_ptr<sycl::access::decorated::no>().get(), binding_capacity,
                            memory(), defer);
                };
                if (!cooperative) {
                    h.single_task<reduction_scalar_kernel>([=]() {
                        outcome status;
                        do { status = step(false); } while (status == outcome::running);
                        out[0] = status;
                        completed_products[0] = 0;
                    });
                } else {
                    h.use_kernel_bundle(*m_bundle);
                    auto c = columns->get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for<reduction_cooperative_kernel>(
                        sycl::nd_range<1>(sycl::range<1>(group_size), sycl::range<1>(group_size)),
                        [=](sycl::nd_item<1> item) {
                            auto group = item.get_group();
                            auto lane = item.get_local_linear_id();
                            auto data = memory();
                            outcome status = outcome::running;
                            std::uint64_t count = 0;
                            while (true) {
                                if (lane == 0 && status == outcome::running) {
                                    do { status = step(true); } while (status == outcome::running);
                                }
                                status = static_cast<outcome>(sycl::group_broadcast(group, static_cast<unsigned>(status), 0));
                                // Publish leader-written frames/values before any
                                // worker reads them. All lanes reach every barrier.
                                sycl::group_barrier(group);
                                if (status != outcome::product_ready) break;
                                auto const & frame = data.m_frames[m[0].m_num_frames - 1];
                                auto left = m[0].view(frame.m_value, data);
                                auto right = m[0].view(m[0].m_value, data);
                                auto size = left.m_size + right.m_size;
                                for (std::size_t column = lane; column < size;) {
                                    c[column] = multiply_natural_column(left, right, column);
                                    if (size - column <= group_size) break;
                                    column += group_size;
                                }
                                // Consume columns only after every writer finishes.
                                sycl::group_barrier(group);
                                if (lane == 0) {
                                    status = m[0].finish_product(data,
                                        c.get_multi_ptr<sycl::access::decorated::no>().get(), size);
                                    if (status == outcome::running && count != UINT64_MAX) ++count;
                                }
                                // Nobody retains reads of the old frame or operands
                                // when the leader advances to the next reduction.
                                sycl::group_barrier(group);
                            }
                            if (lane == 0) { out[0] = status; completed_products[0] = count; }
                        });
                }
            });
            m_queue.wait_and_throw();
        }
        // Buffer destruction has completed host copy-back. Surface any async
        // errors before exposing the candidate state to the session commit.
        m_queue.throw_asynchronous();
        device_arguments.resize(argument_capacity);
        device_bindings.resize(binding_capacity);
        device_values.resize(value_capacity);
        device_frames.resize(frame_capacity);
        arguments.swap(device_arguments);
        bindings.swap(device_bindings);
        arithmetic.m_values.swap(device_values);
        arithmetic.m_frames.swap(device_frames);
        state = next;
        m_last_products = products;
        return result;
    }
};

reduction_sycl_backend::reduction_sycl_backend(reduction_sycl_execution execution):m_execution(execution) {}
reduction_sycl_backend::~reduction_sycl_backend() = default;

std::size_t reduction_sycl_backend::work_group_size() const { return m_imp ? m_imp->work_group_size() : 0; }
std::uint64_t reduction_sycl_backend::last_cooperative_products() const {
    return m_imp ? m_imp->last_cooperative_products() : 0;
}

reduction::outcome reduction_sycl_backend::operator()(
    std::vector<reduction::instruction> const & code, reduction::machine & state,
    std::vector<reduction::closure> & arguments, std::vector<reduction::binding> & bindings,
    reduction::arithmetic_workspace & arithmetic) {
    try {
        // Create the device lazily: an unavailable GPU is an execution outcome,
        // so the installed extension can report it and take the CPU path.
        if (!m_imp) m_imp.reset(new imp(m_execution));
        return (*m_imp)(code, state, arguments, bindings, arithmetic);
    } catch (sycl::exception const & error) {
        throw reduction_backend_error(error.what());
    }
}
}
