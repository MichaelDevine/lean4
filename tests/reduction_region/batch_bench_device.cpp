/* Captured-region throughput, not a whole-proof performance claim.
   Arguments: bits, region count, repetitions, CPU workers (fixture settings). */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include "runtime/init_module.h"
#include "runtime/thread.h"
#include "kernel/type_checker.h"
#include "kernel/reduction_session.h"
#include "library/reduction_sycl.h"
extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);
using namespace lean;
using namespace lean::reduction;

static void require(bool ok, char const * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static expr binary(char const * op, expr const & a, expr const & b) {
    return mk_app(mk_app(mk_const(name({"Nat", op})), a), b);
}

int main(int argc, char ** argv) {
    if (argc > 5) return 2;
    auto bits = argc > 1 ? std::stoull(argv[1]) : 16384;
    auto count = argc > 2 ? std::stoull(argv[2]) : 128;
    auto repeats = argc > 3 ? std::stoull(argv[3]) : 3;
    auto workers = argc > 4 ? std::stoull(argv[4]) : std::max(1u, std::thread::hardware_concurrency());
    require(bits >= 4 && count > 0 && repeats > 0 && workers > 0 && count <= SIZE_MAX / 8,
            "positive fixture sizes and at least four bits are required");
    workers = std::min(workers, count);
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
    nat base(2u), exponent(static_cast<unsigned long>(bits));
    nat value(nat_pow(base.raw(), exponent.raw()));
    require(value > nat(static_cast<unsigned long>(8 * count)), "fixture width is too small for distinct inputs");
    std::vector<expr> inputs, expected;
    inputs.reserve(count); expected.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto literal_at = [&](unsigned offset) {
            return mk_lit(literal(value - nat(static_cast<unsigned long>(8 * i + offset))));
        };
        expr a = literal_at(1), b = literal_at(3), c = literal_at(5), d = literal_at(7);
        inputs.push_back(binary("sub", binary("add", binary("mul", a, b), binary("mul", c, d)), a));
        type_checker checker(env);
        expected.push_back(checker.whnf(inputs.back()));
    }
    // Shared input objects must use Lean's multithread-safe reference counts.
    // Apply the same representation before every timed path, including GPU.
    lean_mark_mt(env.raw());
    for (auto const & input : inputs) lean_mark_mt(input.raw());
    reduction_sycl_backend gpu(reduction_sycl_execution::cooperative, true);
    using clock = std::chrono::steady_clock;
    auto run = [&](unsigned mode, char const * phase, unsigned long long round) {
        auto start = clock::now();
        double prepare_ms = 0, submit_ms = 0, reconstruct_ms = 0;
        std::vector<expr> outputs(count);
        if (mode == 0) {
            for (std::size_t i = 0; i < count; ++i) {
                type_checker checker(env); // fresh cache per input in all CPU paths
                outputs[i] = checker.whnf(inputs[i]);
            }
        } else if (mode == 1) {
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(workers);
            threads.reserve(workers);
            try {
                for (std::size_t worker = 0; worker < workers; ++worker)
                    threads.emplace_back([&, worker]() {
                        lean_initialize_thread();
                        try {
                            for (std::size_t i = worker; i < count; i += workers) {
                                type_checker checker(env);
                                auto result = checker.whnf(inputs[i]);
                                lean_mark_mt(result.raw());
                                outputs[i] = std::move(result);
                            }
                        } catch (...) { errors[worker] = std::current_exception(); }
                        lean_finalize_thread();
                    });
            } catch (...) {
                for (auto & thread : threads) thread.join();
                throw;
            }
            for (auto & thread : threads) thread.join();
            for (auto const & error : errors) if (error) std::rethrow_exception(error);
        } else {
            std::vector<std::unique_ptr<reduction_session>> owners;
            std::vector<reduction_session *> sessions;
            owners.reserve(count); sessions.reserve(count);
            for (auto const & input : inputs) {
                owners.emplace_back(new reduction_session(input, 0, 0, reduction_mode::full(), &env));
                auto & s = *owners.back();
                // Data-derived capacity for the four-operation fixture.
                s.resize_workspace(s.program_size(), 0, s.program_size(),
                                   2 * s.literal_size() + 4, s.literal_size());
                sessions.push_back(&s);
            }
            auto prepared = clock::now();
            auto results = reduction_session::advance_batch(sessions, gpu);
            auto submitted = clock::now();
            for (std::size_t i = 0; i < count; ++i) {
                require(results[i] == outcome::complete, "timed batch failed to complete on GPU");
                outputs[i] = sessions[i]->reconstruct();
            }
            auto reconstructed = clock::now();
            prepare_ms = std::chrono::duration<double, std::milli>(prepared - start).count();
            submit_ms = std::chrono::duration<double, std::milli>(submitted - prepared).count();
            reconstruct_ms = std::chrono::duration<double, std::milli>(reconstructed - submitted).count();
        }
        auto end = clock::now();
        for (std::size_t i = 0; i < count; ++i)
            require(is_bi_equal(outputs[i], expected[i]), "timed batch result differs from CPU reference");
        if (mode == 2)
            require(gpu.last_region_count() == count && gpu.last_cooperative_products() == 2 * count &&
                    gpu.last_kernel_nanoseconds().has_value(),
                    "timed GPU batch did not distribute every region and product");
        std::printf("phase=%s round=%llu mode=%s bits=%llu regions=%llu cpu_workers=%llu elapsed_ms=%.6f prepare_ms=%.6f submit_ms=%.6f reconstruct_ms=%.6f kernel_ms=%.6f\n",
                    phase, round, mode == 0 ? "cpu-serial" : (mode == 1 ? "cpu-parallel" : "gpu-batch"),
                    bits, count, workers, std::chrono::duration<double, std::milli>(end - start).count(),
                    prepare_ms, submit_ms, reconstruct_ms,
                    mode == 2 ? *gpu.last_kernel_nanoseconds() / 1000000.0 : 0);
        std::fflush(stdout);
    };
    for (unsigned mode = 0; mode < 3; ++mode) run(mode, "cold", 0);
    for (unsigned long long round = 0; round < repeats; ++round)
        for (unsigned offset = 0; offset < 3; ++offset) run((round + offset) % 3, "warm", round);
    std::puts("captured-region throughput checks passed");
}
