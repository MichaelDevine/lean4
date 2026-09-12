/* Shared, additive CPU/device arithmetic controls; not a runtime ABI. */
#pragma once
#include <vector>
#include "kernel/reduction_natural.h"

namespace lean {
namespace reduction {

struct natural_job {
    std::size_t m_a, m_a_size, m_b, m_b_size, m_output, m_stride;
};
struct natural_job_result {
    natural_result m_product, m_sum, m_difference;
};

/** One demanded arithmetic chain stays on the executing backend throughout:
    p := a*b; s := p+a; r := s-b. Independent jobs share no writable storage. */
inline natural_job_result run_natural_job(natural_job const & job,
                                          natural_limb const * input, natural_limb * output) {
    natural_view a{input + job.m_a, job.m_a_size}, b{input + job.m_b, job.m_b_size};
    natural_limb * product = output + job.m_output;
    natural_limb * sum = product + job.m_stride;
    natural_limb * difference = sum + job.m_stride;
    natural_result pending{natural_status::invalid, 0};
    auto p = multiply_natural(a, b, product, job.m_stride);
    if (p.m_status != natural_status::complete) return {p, pending, pending};
    auto s = add_natural({product, p.m_size}, a, sum, job.m_stride);
    if (s.m_status != natural_status::complete) return {p, s, pending};
    auto d = subtract_natural({sum, s.m_size}, b, difference, job.m_stride);
    return {p, s, d};
}

inline natural_job_result finish_natural_job(natural_job const & job,
    natural_limb const * input, natural_column const * columns, natural_limb * output) {
    natural_view a{input + job.m_a, job.m_a_size}, b{input + job.m_b, job.m_b_size};
    auto product = output + job.m_output;
    auto sum = product + job.m_stride;
    auto difference = sum + job.m_stride;
    auto size = a.m_size == 0 || b.m_size == 0 ? 0 : a.m_size + b.m_size;
    auto p = finish_natural_product(columns + job.m_output, size, product, job.m_stride);
    natural_result pending{natural_status::invalid, 0};
    if (p.m_status != natural_status::complete) return {p, pending, pending};
    auto s = add_natural({product, p.m_size}, a, sum, job.m_stride);
    if (s.m_status != natural_status::complete) return {p, s, pending};
    auto d = subtract_natural({sum, s.m_size}, b, difference, job.m_stride);
    return {p, s, d};
}

struct natural_gpu_timing { double m_total_ms; double m_kernel_ms; };
natural_gpu_timing run_natural_jobs_gpu(std::vector<natural_limb> const & input,
    std::vector<natural_job> const & jobs, std::vector<natural_limb> & output,
    std::vector<natural_job_result> & results);

}
}
