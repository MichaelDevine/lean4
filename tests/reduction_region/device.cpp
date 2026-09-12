/* Additive SYCL execution test. Requires an actual GPU; no CPU fallback. */
#include <iostream>
#include <sycl/sycl.hpp>
#include "kernel/reduction_machine.h"

int main() {
    using namespace lean::reduction;
    // ((fun x => fun y => x) 11) unsupported: the second argument is unused.
    instruction program[]{
        {opcode::app, 1, 6, 0}, {opcode::app, 2, 5, 1},
        {opcode::lambda, 3, 0, 2}, {opcode::lambda, 4, 0, 3},
        {opcode::bound, 1, 0, 4}, {opcode::value, 0, 0, 11},
        {opcode::unsupported, 0, 0, 6}
    };
    lean::reduction::index result[2]{};
    sycl::queue queue(sycl::gpu_selector_v);
    {
        sycl::buffer<instruction, 1> input(program, sycl::range<1>(7));
        sycl::buffer<lean::reduction::index, 1> output(result, sycl::range<1>(2));
        queue.submit([&](sycl::handler & h) {
            auto code = input.get_access<sycl::access::mode::read>(h);
            auto out = output.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                closure arguments[2]; binding bindings[2]; machine m(0);
                outcome status;
                do {
                    status = m.step(code.get_multi_ptr<sycl::access::decorated::no>().get(),
                                    7, arguments, 2, bindings, 2);
                } while (status == outcome::running);
                out[0] = static_cast<lean::reduction::index>(status);
                out[1] = code[m.m_control.m_code].m_source;
            });
        });
        queue.wait_and_throw();
    }
    if (result[0] != static_cast<lean::reduction::index>(outcome::complete) || result[1] != 11)
        return 1;
    std::cout << "lazy closure executed on "
              << queue.get_device().get_info<sycl::info::device::name>() << '\n';
}
