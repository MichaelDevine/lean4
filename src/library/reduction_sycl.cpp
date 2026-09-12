/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#include <sycl/sycl.hpp>
#include "library/reduction_sycl.h"
#include "kernel/reduction_backend_error.h"

namespace lean {

class reduction_sycl_backend::imp {
    sycl::queue m_queue;
public:
    imp():
        m_queue(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto const & error : errors) std::rethrow_exception(error);
        }) {}

    reduction::outcome operator()(std::vector<reduction::instruction> const & code,
                                  reduction::machine & state,
                                  std::vector<reduction::closure> & arguments,
                                  std::vector<reduction::binding> & bindings,
                                  reduction::arithmetic_workspace & arithmetic) {
        using namespace reduction;
        if (code.empty()) return outcome::invalid;
        auto code_size = code.size();
        auto argument_capacity = arguments.size();
        auto binding_capacity = bindings.size();
        auto literal_count = arithmetic.m_literals.size();
        auto value_capacity = arithmetic.m_values.size();
        auto frame_capacity = arithmetic.m_frames.size();
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
        {
            sycl::buffer<instruction, 1> input(code.data(), sycl::range<1>(code_size));
            sycl::buffer<closure, 1> args(device_arguments.data(), sycl::range<1>(device_arguments.size()));
            sycl::buffer<binding, 1> env(device_bindings.data(), sycl::range<1>(device_bindings.size()));
            sycl::buffer<natural_limb, 1> literals(device_literals.data(), sycl::range<1>(device_literals.size()));
            sycl::buffer<natural_limb, 1> values(device_values.data(), sycl::range<1>(device_values.size()));
            sycl::buffer<arithmetic_frame, 1> frames(device_frames.data(), sycl::range<1>(device_frames.size()));
            sycl::buffer<machine, 1> control(&next, sycl::range<1>(1));
            sycl::buffer<outcome, 1> output(&result, sycl::range<1>(1));
            m_queue.submit([&](sycl::handler & h) {
                auto instructions = input.get_access<sycl::access::mode::read>(h);
                auto a = args.get_access<sycl::access::mode::read_write>(h);
                auto b = env.get_access<sycl::access::mode::read_write>(h);
                auto l = literals.get_access<sycl::access::mode::read>(h);
                auto v = values.get_access<sycl::access::mode::read_write>(h);
                auto f = frames.get_access<sycl::access::mode::read_write>(h);
                auto m = control.get_access<sycl::access::mode::read_write>(h);
                auto out = output.get_access<sycl::access::mode::write>(h);
                h.single_task([=]() {
                    outcome status;
                    do {
                        status = m[0].step(
                            instructions.get_multi_ptr<sycl::access::decorated::no>().get(), code_size,
                            a.get_multi_ptr<sycl::access::decorated::no>().get(), argument_capacity,
                            b.get_multi_ptr<sycl::access::decorated::no>().get(), binding_capacity,
                            {l.get_multi_ptr<sycl::access::decorated::no>().get(), literal_count,
                             v.get_multi_ptr<sycl::access::decorated::no>().get(), value_capacity,
                             f.get_multi_ptr<sycl::access::decorated::no>().get(), frame_capacity});
                    } while (status == outcome::running);
                    out[0] = status;
                });
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
        return result;
    }
};

reduction_sycl_backend::reduction_sycl_backend() = default;
reduction_sycl_backend::~reduction_sycl_backend() = default;

reduction::outcome reduction_sycl_backend::operator()(
    std::vector<reduction::instruction> const & code, reduction::machine & state,
    std::vector<reduction::closure> & arguments, std::vector<reduction::binding> & bindings,
    reduction::arithmetic_workspace & arithmetic) {
    try {
        // Create the device lazily: an unavailable GPU is an execution outcome,
        // so the installed extension can report it and take the CPU path.
        if (!m_imp) m_imp.reset(new imp());
        return (*m_imp)(code, state, arguments, bindings, arithmetic);
    } catch (sycl::exception const & error) {
        throw reduction_backend_error(error.what());
    }
}
}
