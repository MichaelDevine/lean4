/* Additive actual-GPU control. No Lean host object enters device compilation. */
#include <chrono>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "natural_fixture.h"

namespace lean {
namespace reduction {

natural_gpu_timing run_natural_jobs_gpu(std::vector<natural_limb> const & input,
    std::vector<natural_job> const & jobs, std::vector<natural_limb> & output,
    std::vector<natural_job_result> & results) {
    if (jobs.empty() || input.empty() || output.empty() || results.size() != jobs.size())
        throw std::invalid_argument("invalid GPU arithmetic control shape");
    auto start = std::chrono::steady_clock::now();
    // A fixture-local persistent queue permits a second, warm measurement.
    // Production device/session ownership is deliberately not defined here.
    static thread_local sycl::queue queue(sycl::gpu_selector_v, [](sycl::exception_list errors) {
        for (auto const & error : errors) std::rethrow_exception(error);
    }, sycl::property::queue::enable_profiling());
    sycl::event columns_event, submitted;
    // The fixture reuses output offsets for column workspace. Real admission
    // may pack it more tightly; no allocation is hidden inside the kernel.
    std::size_t width = 0;
    for (auto const & job : jobs) {
        if (job.m_a_size + job.m_b_size > width) width = job.m_a_size + job.m_b_size;
    }
    {
        sycl::buffer<natural_limb, 1> in(input.data(), sycl::range<1>(input.size()));
        sycl::buffer<natural_job, 1> work(jobs.data(), sycl::range<1>(jobs.size()));
        sycl::buffer<natural_limb, 1> out(output.data(), sycl::range<1>(output.size()));
        sycl::buffer<natural_job_result, 1> result(results.data(), sycl::range<1>(results.size()));
        sycl::buffer<natural_column, 1> columns(sycl::range<1>(output.size()));
        columns_event = queue.submit([&](sycl::handler & h) {
            auto a = in.get_access<sycl::access::mode::read>(h);
            auto w = work.get_access<sycl::access::mode::read>(h);
            auto c = columns.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<2>(jobs.size(), width), [=](sycl::id<2> i) {
                auto job = w[i[0]];
                auto column = i[1];
                if (column < job.m_a_size + job.m_b_size) {
                    auto data = a.get_multi_ptr<sycl::access::decorated::no>().get();
                    c[job.m_output + column] = multiply_natural_column(
                        {data + job.m_a, job.m_a_size}, {data + job.m_b, job.m_b_size}, column);
                }
            });
        });
        submitted = queue.submit([&](sycl::handler & h) {
            auto a = in.get_access<sycl::access::mode::read>(h);
            auto w = work.get_access<sycl::access::mode::read>(h);
            auto b = out.get_access<sycl::access::mode::read_write>(h);
            auto r = result.get_access<sycl::access::mode::write>(h);
            auto c = columns.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(jobs.size()), [=](sycl::id<1> i) {
                r[i] = finish_natural_job(w[i],
                    a.get_multi_ptr<sycl::access::decorated::no>().get(),
                    c.get_multi_ptr<sycl::access::decorated::no>().get(),
                    b.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
        queue.wait_and_throw();
    }
    queue.throw_asynchronous();
    auto end = std::chrono::steady_clock::now();
    auto begin_kernel = columns_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_kernel = submitted.get_profiling_info<sycl::info::event_profiling::command_end>();
    return {std::chrono::duration<double, std::milli>(end - start).count(),
            static_cast<double>(end_kernel - begin_kernel) / 1000000.0};
}

}
}
