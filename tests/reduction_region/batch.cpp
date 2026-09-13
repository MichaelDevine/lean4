/* Additive independent-region batching, isolation and recovery controls. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#include "kernel/reduction_session.h"
#include "kernel/reduction_backend_error.h"
#ifdef LEAN_REDUCTION_BATCH_GPU
#include "library/reduction_sycl.h"
#endif
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
static void provision(reduction_session & s) {
    // Bounds for these fixed fixture expressions, not an admission default.
    s.resize_workspace(s.program_size(), s.program_size(), s.program_size(),
                       6 * s.literal_size() + 4, 2 * s.literal_size());
}
template<typename Backend> struct counting_batch {
    Backend & m_backend;
    unsigned m_calls = 0;
    std::vector<outcome> submit(std::vector<submission> & requests) {
        ++m_calls;
        return m_backend.submit(requests);
    }
};
template<typename Backend> struct failing_batch {
    Backend & m_backend;
    std::vector<outcome> submit(std::vector<submission> & requests) {
        m_backend.submit(requests);
        throw reduction_backend_error("injected failure after candidate computation");
    }
};
template<typename Backend> struct invalid_input_batch {
    Backend & m_backend;
    std::vector<outcome> submit(std::vector<submission> & requests) {
        requests[0].m_state.m_control.m_code = UINT64_MAX;
        return m_backend.submit(requests);
    }
};
template<typename Backend> struct invalid_count_batch {
    Backend & m_backend;
    unsigned m_case;
    std::vector<outcome> submit(std::vector<submission> & requests) {
        auto & r = requests[0];
        if (m_case == 0) r.m_state.m_num_arguments = r.m_arguments.size() + 1;
        if (m_case == 1) r.m_state.m_num_bindings = r.m_bindings.size() + 1;
        if (m_case == 2) r.m_state.m_num_values = r.m_arithmetic.m_values.size() + 1;
        if (m_case == 3) r.m_state.m_num_frames = r.m_arithmetic.m_frames.size() + 1;
        return m_backend.submit(requests);
    }
};
struct malformed_batch {
    unsigned m_case;
    std::vector<outcome> submit(std::vector<submission> & requests) {
        reduction_cpu_backend cpu;
        auto results = cpu.submit(requests);
        if (m_case == 0) results.pop_back();
        if (m_case == 1) requests[0].m_state.m_control.m_code = UINT64_MAX;
        if (m_case == 2) results[0] = outcome::product_ready;
        return results;
    }
};

template<typename Backend> void run_controls(Backend & backend, environment const & env) {
    std::vector<submission> empty;
    require(backend.submit(empty).empty(), "empty backend batch produced an outcome");
#ifdef LEAN_REDUCTION_BATCH_GPU
    require(backend.work_group_size() == 0 && backend.last_region_count() == 0,
            "empty batch initialized the GPU");
#endif
    expr type = mk_const(name("Nat")), one = mk_lit(literal(1u));
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr unused = mk_mvar(name("unused"));
    std::vector<expr> inputs, expected;
    std::vector<std::unique_ptr<reduction_session>> owners;
    std::vector<reduction_session *> sessions;
    unsigned widths[]{0, 1, 31, 32, 33, 127, 128, 129, 511, 1024, 4096};
    // Different code lengths, literal lengths, bindings and zero-sized spans
    // catch accidental global rather than region-relative indexing.
    for (unsigned i = 0; i < 73; ++i) {
        nat base(2u), exponent(widths[i % 11]);
        nat value(nat_pow(base.raw(), exponent.raw()));
        expr a = mk_lit(literal(value - nat(1u))), b = mk_lit(literal(nat(i)));
        expr input = binary("sub", binary("add", binary("mul", a, b), a), b);
        if (i % 4 == 1) input = mk_app(identity, input);
        if (i % 4 == 2) input = mk_let(name("x"), type, input, mk_bvar(0), false);
        if (i % 4 == 3) input = mk_let(name("unused"), type, unused, input, false);
        inputs.push_back(input);
        type_checker checker(env);
        expected.push_back(checker.whnf(input));
        owners.emplace_back(new reduction_session(input, 0, 0, reduction_mode::full(), &env));
        provision(*owners.back());
        sessions.push_back(owners.back().get());
    }
    // A closed literal needs no mutable storage at all in full mode.
    inputs.push_back(one); expected.push_back(one);
    owners.emplace_back(new reduction_session(one, 0, 0, reduction_mode::full(), &env));
    sessions.push_back(owners.back().get());
    counting_batch counted{backend};
    auto results = reduction_session::advance_batch(sessions, counted);
    require(counted.m_calls == 1 && results.size() == sessions.size(), "regions were not submitted together");
    for (std::size_t i = 0; i < sessions.size(); ++i)
        require(results[i] == outcome::complete && is_bi_equal(sessions[i]->reconstruct(), expected[i]),
                "batched expression differs from its CPU result");
#ifdef LEAN_REDUCTION_BATCH_GPU
    require(backend.last_region_count() == sessions.size(), "GPU did not receive the complete batch");
#ifndef LEAN_REDUCTION_BATCH_SCALAR
    require(backend.work_group_size() > 1 && backend.last_cooperative_products() > 1,
            "batch did not use cooperative arithmetic");
#endif
#endif

    // A large sibling can save a resource continuation while another finishes
    // and a third stops at an unsupported boundary. No global status couples them.
    expr hard = binary("mul", inputs[10], inputs[9]);
    reduction_session suspended(hard, 0, 0, reduction_mode::full(), &env);
    reduction_session ready(one, 0, 0, reduction_mode::full(), &env);
    reduction_session unsupported(unused, 0, 0, reduction_mode::full(), &env);
    std::vector<reduction_session *> mixed{&suspended, &ready, &unsupported};
    results = reduction_session::advance_batch(mixed, counted);
    require(results[0] == outcome::need_arguments && results[1] == outcome::complete &&
            results[2] == outcome::unsupported, "one region's stop changed its siblings' outcomes");
    require(ready.reconstruct() == one && unsupported.reconstruct() == unused, "sibling result changed");
    auto complete_state = ready.state();
    bool saw_frames = false, saw_values = false, saw_columns = false;
    while (results[0] != outcome::complete) {
        auto status = results[0];
        require(status == outcome::need_arguments || status == outcome::need_bindings ||
                status == outcome::need_frames || status == outcome::need_values || status == outcome::need_columns,
                "unexpected batched recovery status");
        saw_frames |= status == outcome::need_frames;
        saw_values |= status == outcome::need_values;
        saw_columns |= status == outcome::need_columns;
        suspended.resize_workspace(
            suspended.argument_capacity() + (status == outcome::need_arguments),
            suspended.binding_capacity() + (status == outcome::need_bindings),
            suspended.frame_capacity() + (status == outcome::need_frames),
            std::max<reduction::index>(suspended.value_capacity(), suspended.state().m_required_values),
            std::max<reduction::index>(suspended.column_capacity(), suspended.state().m_required_columns));
        results = reduction_session::advance_batch(std::vector<reduction_session *>{&suspended}, counted);
    }
    type_checker checker(env);
    require(saw_frames && saw_values && is_bi_equal(suspended.reconstruct(), checker.whnf(hard)),
            "batched resource continuation changed the answer");
#if defined(LEAN_REDUCTION_BATCH_GPU) && !defined(LEAN_REDUCTION_BATCH_SCALAR)
    require(saw_columns, "cooperative batch scratch recovery was not exercised");
#else
    require(!saw_columns, "scalar batch required cooperative scratch");
#endif
    require(ready.state().m_num_values == complete_state.m_num_values && ready.reconstruct() == one,
            "completed sibling was changed during continuation");

    // A backend exception commits none of the computed candidate states.
    reduction_session first(inputs[10], 0, 0, reduction_mode::full(), &env);
    reduction_session second(inputs[9], 0, 0, reduction_mode::full(), &env);
    provision(first); provision(second);
    expr before_first = first.reconstruct(), before_second = second.reconstruct();
    std::vector<reduction_session *> pair{&first, &second};
    failing_batch fail{backend};
    bool caught = false;
    try { reduction_session::advance_batch(pair, fail); } catch (reduction_backend_error const &) { caught = true; }
    require(caught && is_bi_equal(first.reconstruct(), before_first) &&
            is_bi_equal(second.reconstruct(), before_second), "batch exception partially committed");
    results = reduction_session::advance_batch(pair, counted);
    require(results[0] == outcome::complete && results[1] == outcome::complete &&
            is_bi_equal(first.reconstruct(), expected[10]) && is_bi_equal(second.reconstruct(), expected[9]),
            "batch retry after failed transfer changed a result");

    // Reject an invalid region inside the actual device machine without
    // changing the adjacent region's offsets, memory or result.
    reduction_session bad_input(inputs[10], 0, 0, reduction_mode::full(), &env);
    reduction_session good_input(inputs[9], 0, 0, reduction_mode::full(), &env);
    provision(bad_input); provision(good_input);
    invalid_input_batch corrupt{backend};
    results = reduction_session::advance_batch(
        std::vector<reduction_session *>{&bad_input, &good_input}, corrupt);
    require(results[0] == outcome::invalid && is_bi_equal(bad_input.reconstruct(), inputs[10]) &&
            results[1] == outcome::complete && is_bi_equal(good_input.reconstruct(), expected[9]),
            "invalid device input contaminated its sibling or committed");

    for (unsigned fault = 0; fault < 4; ++fault) {
        reduction_session a(inputs[10], 0, 0, reduction_mode::full(), &env);
        reduction_session b(inputs[9], 0, 0, reduction_mode::full(), &env);
        provision(a); provision(b);
        invalid_count_batch corrupt_count{backend, fault};
        results = reduction_session::advance_batch(std::vector<reduction_session *>{&a, &b}, corrupt_count);
        require(results[0] == outcome::invalid && is_bi_equal(a.reconstruct(), inputs[10]) &&
                results[1] == outcome::complete && is_bi_equal(b.reconstruct(), expected[9]),
                "invalid live counter was copied or contaminated its sibling");
    }

    for (unsigned fault = 0; fault < 3; ++fault) {
        reduction_session a(inputs[10], 0, 0, reduction_mode::full(), &env);
        reduction_session b(inputs[9], 0, 0, reduction_mode::full(), &env);
        provision(a); provision(b);
        malformed_batch bad{fault};
        results = reduction_session::advance_batch(std::vector<reduction_session *>{&a, &b}, bad);
        require(results[0] == outcome::invalid && is_bi_equal(a.reconstruct(), inputs[10]),
                "malformed batch result was committed");
        require(fault == 0 ? (results[1] == outcome::invalid && is_bi_equal(b.reconstruct(), inputs[9])) :
                            (results[1] == outcome::complete && is_bi_equal(b.reconstruct(), expected[9])),
                "malformed result isolation failed");
    }
    auto before_calls = counted.m_calls;
    require(reduction_session::advance_batch(std::vector<reduction_session *>{}, counted).empty() &&
            counted.m_calls == before_calls, "empty batch dispatched work");
    for (auto invalid : {std::vector<reduction_session *>{&first, &first},
                         std::vector<reduction_session *>{&first, nullptr}}) {
        bool rejected = false;
        try { reduction_session::advance_batch(invalid, counted); } catch (std::invalid_argument const &) { rejected = true; }
        require(rejected && counted.m_calls == before_calls, "invalid ownership reached the backend");
    }
#if defined(LEAN_REDUCTION_BATCH_GPU) && !defined(LEAN_REDUCTION_BATCH_SCALAR)
    reduction_session overflow(one, 0, 0, reduction_mode::full(), &env);
    overflow.resize_workspace(0, 0, 0, 0, SIZE_MAX);
    bool rejected = false;
    try { reduction_session::advance_batch(std::vector<reduction_session *>{&overflow}, counted); }
    catch (std::length_error const &) { rejected = true; }
    require(rejected && overflow.reconstruct() == one && overflow.state().m_num_values == 0,
            "scratch size overflow was allocated or committed");
#endif
    require(backend.submit(empty).empty(), "empty reused backend returned stale outcomes");
#ifdef LEAN_REDUCTION_BATCH_GPU
    require(backend.last_region_count() == 0 && backend.last_cooperative_products() == 0,
            "empty reused backend retained stale telemetry");
    require(!backend.last_kernel_nanoseconds().has_value(), "disabled profiling produced a timestamp");
#endif
    std::printf("independent region batch checks passed: %zu; mixed_recovery=yes; transactional=yes\n", sessions.size());
}

#ifdef LEAN_REDUCTION_BATCH_GPU
static void run_storage_controls(reduction_sycl_backend & backend, environment const & env) {
    nat base(2u), exponent(4096u);
    nat value(nat_pow(base.raw(), exponent.raw()));
    auto input = [&](unsigned offset) {
        auto a = mk_lit(literal(value - nat(offset)));
        return binary("add", binary("mul", a, a), a);
    };
    auto evaluate = [&](expr const & root, std::size_t spare) {
        reduction_session s(root, 0, 0, reduction_mode::full(), &env);
        provision(s);
        s.resize_workspace(s.argument_capacity() + spare, s.binding_capacity() + spare,
                           s.frame_capacity() + spare, s.value_capacity() + spare, s.column_capacity() + spare);
        type_checker checker(env);
        auto result = s.advance(backend);
        require(result == outcome::complete && is_bi_equal(s.reconstruct(), checker.whnf(root)),
                "storage reuse changed a newly captured expression");
        return std::make_pair(backend.last_explicit_upload_bytes(), backend.last_explicit_download_bytes());
    };
    auto original = evaluate(input(1), 0);
    auto small_bytes = backend.retained_device_bytes();
    // A large unused tail must not become transferred live data. Changing the
    // fresh input at equal dimensions catches stale device-content reuse.
    auto padded = evaluate(input(3), 65536);
    auto large_bytes = backend.retained_device_bytes();
    if (backend.uses_resident_storage()) {
        require(original.first > 0 && original.second > 0 && original == padded,
                "resident copies scale with unused capacity rather than live data");
        require(large_bytes > small_bytes, "resident workspace growth was not recorded");
        evaluate(input(5), 65536);
        require(backend.retained_device_bytes() == large_bytes, "same-shape work grew retained allocations");
        evaluate(input(7), 0);
        require(backend.retained_device_bytes() == large_bytes, "smaller work did not reuse retained allocations");
    } else {
        require(original == std::make_pair(std::size_t(0), std::size_t(0)) && small_bytes == 0 && large_bytes == 0,
                "buffer transport claimed explicit resident bytes");
    }
#ifdef LEAN_REDUCTION_BATCH_RESIDENT
    require(backend.uses_resident_storage(), "explicit resident control silently changed transport");
#endif
#ifdef LEAN_REDUCTION_BATCH_BUFFERS
    require(!backend.uses_resident_storage(), "explicit buffer control silently changed transport");
#endif
    reduction_session suspended(input(9), 0, 0, reduction_mode::full(), &env);
    require(suspended.advance(backend) == outcome::need_arguments, "release control did not save a continuation");
    auto width = backend.work_group_size();
    backend.release_storage();
    require(backend.retained_device_bytes() == 0 && backend.work_group_size() == width,
            "release retained workspace or discarded the initialized kernel");
    backend.release_storage(); // idle release is repeatable
    provision(suspended);
    type_checker checker(env);
    require(suspended.advance(backend) == outcome::complete &&
            is_bi_equal(suspended.reconstruct(), checker.whnf(input(9))),
            "releasing device storage lost an independent host checkpoint");
}
#endif

int main() {
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    require(lean_io_result_is_ok(created), "environment creation failed");
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
#ifdef LEAN_REDUCTION_BATCH_GPU
#ifdef LEAN_REDUCTION_BATCH_SCALAR
    reduction_sycl_backend backend(reduction_sycl_execution::scalar);
#elif defined(LEAN_REDUCTION_BATCH_RESIDENT)
    reduction_sycl_backend backend(reduction_sycl_execution::cooperative, false, reduction_sycl_storage::resident);
#elif defined(LEAN_REDUCTION_BATCH_BUFFERS)
    reduction_sycl_backend backend(reduction_sycl_execution::cooperative, false, reduction_sycl_storage::buffers);
#else
    reduction_sycl_backend backend;
#endif
#else
    reduction_cpu_backend backend;
#endif
    run_controls(backend, env);
#ifdef LEAN_REDUCTION_BATCH_GPU
    run_storage_controls(backend, env);
#endif
}
