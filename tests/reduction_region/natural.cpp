/* Additive exact Nat arithmetic checks against Lean's selected CPU runtime. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "runtime/init_module.h"
#include "util/nat.h"
#include "natural_fixture.h"

extern "C" void lean_initialize();
using namespace lean;
using namespace lean::reduction;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

// Reference conversion uses Lean's arbitrary-precision operations; it does
// not use the new limb arithmetic or reinterpret a GMP internal allocation.
static nat from_limbs(natural_limb const * data, std::size_t size) {
    nat value;
    nat base("4294967296");
    for (std::size_t i = size; i != 0; --i) value = value * base + nat(data[i - 1]);
    return value;
}

static void resource_controls() {
    natural_limb max[] = {UINT32_MAX, UINT32_MAX};
    natural_limb one[] = {1};
    natural_limb bad[] = {5, 0};
    std::vector<natural_limb> storage(8, 0xabc123);
    auto original = storage;
    auto a = add_natural({max, 2}, {one, 1}, storage.data(), 2);
    auto b = subtract_natural({max, 2}, {one, 1}, storage.data(), 1);
    auto c = multiply_natural({max, 2}, {max, 2}, storage.data(), 3);
    require(a.m_status == natural_status::need_space && a.m_size == 3 &&
            b.m_status == natural_status::need_space && b.m_size == 2 &&
            c.m_status == natural_status::need_space && c.m_size == 4 && storage == original,
            "resource decline modified destination or lost the required capacity");
    require(add_natural({bad, 2}, {one, 1}, storage.data(), storage.size()).m_status == natural_status::invalid &&
            multiply_natural({nullptr, 1}, {one, 1}, storage.data(), storage.size()).m_status == natural_status::invalid &&
            subtract_natural({one, 1}, {bad, 2}, storage.data(), storage.size()).m_status == natural_status::invalid &&
            storage == original, "invalid representation modified destination");
    require(add_natural({nullptr, 0}, {nullptr, 0}, nullptr, 0).m_size == 0 &&
            multiply_natural({nullptr, 0}, {max, 2}, nullptr, 0).m_size == 0 &&
            subtract_natural({one, 1}, {max, 2}, nullptr, 0).m_size == 0,
            "zero or saturated result unnecessarily demanded workspace");
    a = add_natural({max, 2}, {one, 1}, storage.data(), 3);
    require(a.m_status == natural_status::complete && a.m_size == 3 &&
            storage[0] == 0 && storage[1] == 0 && storage[2] == 1,
            "retry after growth lost a carry");
    natural_limb power[] = {0, 0, 1};
    b = subtract_natural({power, 3}, {one, 1}, storage.data(), 3);
    require(b.m_status == natural_status::complete && b.m_size == 2 &&
            storage[0] == UINT32_MAX && storage[1] == UINT32_MAX,
            "borrow chain or normalization failed");
    natural_column columns[] = {{UINT64_MAX, UINT64_MAX}, {0, 0}};
    require(finish_natural_product(columns, 2, storage.data(), 1).m_status == natural_status::need_space,
            "column finishing ignored insufficient capacity");
    require(finish_natural_product(columns, 2, storage.data(), 2).m_status == natural_status::invalid,
            "column finishing truncated a residual carry");
    for (unsigned x = 0; x < 64; ++x) {
        for (unsigned y = 0; y < 64; ++y) {
            natural_limb u[] = {x}, v[] = {y};
            natural_view left{u, x != 0}, right{v, y != 0};
            auto sum = add_natural(left, right, storage.data(), storage.size());
            require(sum.m_status == natural_status::complete &&
                    from_limbs(storage.data(), sum.m_size) == nat(x) + nat(y), "small addition mismatch");
            auto sub = subtract_natural(left, right, storage.data(), storage.size());
            require(sub.m_status == natural_status::complete &&
                    from_limbs(storage.data(), sub.m_size) == nat(x) - nat(y), "small subtraction mismatch");
            auto mul = multiply_natural(left, right, storage.data(), storage.size());
            require(mul.m_status == natural_status::complete &&
                    from_limbs(storage.data(), mul.m_size) == nat(x) * nat(y), "small multiplication mismatch");
        }
    }
}

static int natural_controls(bool device) {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    resource_controls();
    std::mt19937_64 random(0xb504337);
    std::vector<natural_limb> input;
    std::vector<natural_job> jobs;
    std::size_t output_size = 0;
    auto add_job = [&](std::size_t as, std::size_t bs, unsigned pattern) {
        auto a = input.size();
        for (std::size_t i = 0; i < as; ++i)
            input.push_back(pattern == 0 ? UINT32_MAX : static_cast<natural_limb>(random()));
        if (as != 0 && input.back() == 0) input.back() = 1;
        auto b = input.size();
        for (std::size_t i = 0; i < bs; ++i)
            input.push_back(pattern == 0 ? UINT32_MAX : static_cast<natural_limb>(random()));
        if (bs != 0 && input.back() == 0) input.back() = 1;
        // Explicit test data dimensions, not a production precision ceiling.
        auto stride = as + bs + 2;
        jobs.push_back({a, as, b, bs, output_size, stride});
        output_size += stride * 3 + 1; // one untouched guard per job
    };
    for (std::size_t a : {0, 1, 2, 3, 4, 31, 32, 33, 127, 128})
        for (std::size_t b : {0, 1, 2, 3, 7, 32, 65, 129})
            for (unsigned p = 0; p < 2; ++p) add_job(a, b, p);
    for (unsigned i = 0; i < 1024; ++i) add_job(random() % 129, random() % 129, 1);
    for (unsigned i = 0; i < 4096; ++i) add_job(128, 128, 1);
    add_job(1024, 1024, 0); // a 32768-bit carry-heavy product
    add_job(1024, 1, 1);
    std::vector<natural_limb> output(output_size, 0x123abc);
    std::vector<natural_job_result> results(jobs.size());
    std::vector<nat> expected;
    using clock = std::chrono::steady_clock;
    double reference_ms = 0;
    for (auto const & job : jobs) {
        auto a = from_limbs(input.data() + job.m_a, job.m_a_size);
        auto b = from_limbs(input.data() + job.m_b, job.m_b_size);
        auto start = clock::now();
        nat product = a * b, sum = product + a, difference = sum - b;
        auto end = clock::now();
        reference_ms += std::chrono::duration<double, std::milli>(end - start).count();
        expected.push_back(product); expected.push_back(sum); expected.push_back(difference);
    }
    auto start = clock::now();
    double kernel_ms = 0, total_ms = 0;
    if (device) {
#ifdef LEAN_REDUCTION_NATURAL_DEVICE_CONTROL
        auto timing = run_natural_jobs_gpu(input, jobs, output, results);
        total_ms = timing.m_total_ms; kernel_ms = timing.m_kernel_ms;
#else
        require(false, "GPU arithmetic control was not compiled");
#endif
    } else {
        for (std::size_t i = 0; i < jobs.size(); ++i)
            results[i] = run_natural_job(jobs[i], input.data(), output.data());
        total_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();
        // Independently compare the column path with the schoolbook path on
        // CPU as well as comparing their actual GPU results with Lean Nat.
        std::vector<natural_column> columns(output_size);
        std::vector<natural_limb> parallel_output(output_size, 0x123abc);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            auto const & job = jobs[i];
            natural_view a{input.data() + job.m_a, job.m_a_size};
            natural_view b{input.data() + job.m_b, job.m_b_size};
            for (std::size_t c = 0; c < a.m_size + b.m_size; ++c)
                columns[job.m_output + c] = multiply_natural_column(a, b, c);
            auto got = finish_natural_job(job, input.data(), columns.data(), parallel_output.data());
            natural_result values[] = {got.m_product, got.m_sum, got.m_difference};
            natural_result scalar[] = {results[i].m_product, results[i].m_sum, results[i].m_difference};
            for (unsigned j = 0; j < 3; ++j) {
                require(values[j].m_status == scalar[j].m_status && values[j].m_size == scalar[j].m_size,
                        "column and scalar outcomes differ");
                auto offset = job.m_output + j * job.m_stride;
                require(std::equal(output.begin() + offset, output.begin() + offset + scalar[j].m_size,
                                   parallel_output.begin() + offset), "column and scalar arithmetic differ");
            }
        }
    }
    auto validate = [&]() {
        std::size_t checks = 0;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            auto const & job = jobs[i];
            natural_result values[] = {results[i].m_product, results[i].m_sum, results[i].m_difference};
            for (unsigned j = 0; j < 3; ++j) {
                auto r = values[j];
                require(r.m_status == natural_status::complete && r.m_size <= job.m_stride,
                        "arithmetic operation did not complete");
                auto data = output.data() + job.m_output + j * job.m_stride;
                require(valid_natural({data, r.m_size}) && from_limbs(data, r.m_size) == expected[3 * i + j],
                        "portable exact arithmetic differs from Lean CPU arithmetic");
                ++checks;
            }
            require(output[job.m_output + 3 * job.m_stride] == 0x123abc, "job wrote outside its workspace");
        }
        return checks;
    };
    auto checks = validate();
    std::printf("backend=%s jobs=%zu exact_checks=%zu reference_arithmetic_ms=%.3f backend_total_ms=%.3f kernel_ms=%.3f\n",
        device ? "gpu" : "cpu", jobs.size(), checks, reference_ms, total_ms, kernel_ms);
#ifdef LEAN_REDUCTION_NATURAL_DEVICE_CONTROL
    // The warm run must overwrite fresh sentinels and pass the same exact
    // checks; cached output cannot satisfy this second invocation.
    std::fill(output.begin(), output.end(), 0x123abc);
    natural_result invalid{natural_status::invalid, 0};
    std::fill(results.begin(), results.end(), natural_job_result{invalid, invalid, invalid});
    auto warm = run_natural_jobs_gpu(input, jobs, output, results);
    require(validate() == checks, "warm arithmetic check count changed");
    std::printf("backend=gpu-warm jobs=%zu exact_checks=%zu backend_total_ms=%.3f kernel_ms=%.3f\n",
        jobs.size(), checks, warm.m_total_ms, warm.m_kernel_ms);
#endif
    std::puts("exact natural arithmetic controls passed");
    // This fixture submits arithmetic chains, not actual Expr reductions.
    // It is not an integrated checker benchmark or a whole-proof speedup.
    return 0;
}

#ifndef LEAN_REDUCTION_NATURAL_DEVICE_CONTROL
int main() { return natural_controls(false); }
#endif
