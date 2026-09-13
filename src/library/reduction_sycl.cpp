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
class reduction_resident_scalar_kernel;
class reduction_resident_cooperative_kernel;
class reduction_transfer_kernel;

namespace {
struct buffer_slice { std::size_t m_offset, m_size; };
struct region_layout {
    buffer_slice m_code, m_arguments, m_bindings, m_literals, m_values, m_frames, m_columns;
};
struct live_layout { buffer_slice m_arguments, m_bindings, m_values, m_frames; };

/** One device-only view, shared by buffer and resident transports. The local
    indices of every region retain the existing machine's exact semantics. */
struct device_batch {
    region_layout const * m_layout;
    reduction::instruction const * m_code;
    reduction::closure * m_arguments;
    reduction::binding * m_bindings;
    reduction::natural_limb const * m_literals;
    reduction::natural_limb * m_values;
    reduction::arithmetic_frame * m_frames;
    reduction::natural_column * m_columns;
    reduction::machine * m_states;
    reduction::outcome * m_results;
    std::uint64_t * m_products;

    reduction::arithmetic_memory memory(std::size_t region) const {
        auto const & s = m_layout[region];
        return {m_literals + s.m_literals.m_offset, s.m_literals.m_size,
                m_values + s.m_values.m_offset, s.m_values.m_size,
                m_frames + s.m_frames.m_offset, s.m_frames.m_size, s.m_columns.m_size};
    }
    reduction::outcome step(std::size_t region, bool defer) const {
        auto const & s = m_layout[region];
        return m_states[region].step(m_code + s.m_code.m_offset, s.m_code.m_size,
            m_arguments + s.m_arguments.m_offset, s.m_arguments.m_size,
            m_bindings + s.m_bindings.m_offset, s.m_bindings.m_size, memory(region), defer);
    }
};

/** Coalesced transport is separate from the evaluator. One bulk transfer per
    typed live arena replaces one host command per region; the device scatters
    or gathers those prefixes at their independently reserved offsets. */
struct device_transfer {
    live_layout const * m_layout;
    reduction::closure * m_arguments;
    reduction::binding * m_bindings;
    reduction::natural_limb * m_values;
    reduction::arithmetic_frame * m_frames;
};

template<typename T>
void transfer_prefix(T * live, T * arena, buffer_slice prefix, buffer_slice allocation,
                     bool gather, std::size_t lane, std::size_t width) {
    for (auto i = lane; i < prefix.m_size;) {
        if (gather) live[prefix.m_offset + i] = arena[allocation.m_offset + i];
        else arena[allocation.m_offset + i] = live[prefix.m_offset + i];
        if (prefix.m_size - i <= width) break;
        i += width;
    }
}

void transfer_region(device_transfer live, device_batch arena, bool gather, sycl::nd_item<1> item) {
    auto region = item.get_group_linear_id();
    auto lane = item.get_local_linear_id(), width = item.get_local_range(0);
    auto const & p = live.m_layout[region]; auto const & a = arena.m_layout[region];
    transfer_prefix(live.m_arguments, arena.m_arguments, p.m_arguments, a.m_arguments, gather, lane, width);
    transfer_prefix(live.m_bindings, arena.m_bindings, p.m_bindings, a.m_bindings, gather, lane, width);
    transfer_prefix(live.m_values, arena.m_values, p.m_values, a.m_values, gather, lane, width);
    transfer_prefix(live.m_frames, arena.m_frames, p.m_frames, a.m_frames, gather, lane, width);
}

void run_scalar(device_batch view, std::size_t region) {
    reduction::outcome status;
    do { status = view.step(region, false); } while (status == reduction::outcome::running);
    view.m_results[region] = status;
    view.m_products[region] = 0;
}

void run_cooperative(device_batch view, sycl::nd_item<1> item) {
    using namespace reduction;
    auto group = item.get_group();
    auto lane = item.get_local_linear_id();
    auto group_size = item.get_local_range(0);
    auto region = item.get_group_linear_id();
    auto data = view.memory(region);
    auto columns = view.m_columns + view.m_layout[region].m_columns.m_offset;
    auto & state = view.m_states[region];
    outcome status = outcome::running;
    std::uint64_t count = 0;
    while (true) {
        if (lane == 0 && status == outcome::running)
            do { status = view.step(region, true); } while (status == outcome::running);
        status = static_cast<outcome>(sycl::group_broadcast(group, static_cast<unsigned>(status), 0));
        // Every lane follows the same exits and barriers. Publish leader state
        // before reading operands; consume columns after all writers finish.
        sycl::group_barrier(group);
        if (status != outcome::product_ready) break;
        auto const & frame = data.m_frames[state.m_num_frames - 1];
        auto left = state.view(frame.m_value, data), right = state.view(state.m_value, data);
        auto size = left.m_size + right.m_size;
        for (std::size_t column = lane; column < size;) {
            columns[column] = multiply_natural_column(left, right, column);
            if (size - column <= group_size) break;
            column += group_size;
        }
        sycl::group_barrier(group);
        if (lane == 0) {
            status = state.finish_product(data, columns, size);
            if (status == outcome::running && count != UINT64_MAX) ++count;
        }
        // No old-frame read overlaps the next leader reduction.
        sycl::group_barrier(group);
    }
    if (lane == 0) { view.m_results[region] = status; view.m_products[region] = count; }
}

/** Scratch ownership only: no previous contents are a trusted checkpoint.
    The queue outlives its allocations; submitted accesses are drained before
    growth/destruction. Growth may discard the old idle allocation first. */
template<typename T> class device_allocation {
    sycl::queue & m_queue;
    T * m_data = nullptr;
    std::size_t m_capacity = 0;
public:
    explicit device_allocation(sycl::queue & q):m_queue(q) {}
    ~device_allocation() { if (m_data) { try { sycl::free(m_data, m_queue); } catch (...) {} } }
    device_allocation(device_allocation const &) = delete;
    device_allocation & operator=(device_allocation const &) = delete;
    T * data() const { return m_data; }
    std::size_t bytes() const { return m_capacity * sizeof(T); }
    void clear() {
        if (m_data) sycl::free(m_data, m_queue);
        m_data = nullptr; m_capacity = 0;
    }
    void reserve(std::size_t count) {
        count = std::max<std::size_t>(1, count);
        if (count > SIZE_MAX / sizeof(T)) throw std::length_error("device allocation size overflow");
        if (count <= m_capacity) return;
        clear();
        auto p = sycl::malloc_device<T>(count, m_queue);
        if (!p) throw std::bad_alloc();
        m_data = p; m_capacity = count;
    }
};

struct resident_storage {
    device_allocation<region_layout> m_layout;
    device_allocation<reduction::instruction> m_code;
    device_allocation<reduction::closure> m_arguments;
    device_allocation<reduction::binding> m_bindings;
    device_allocation<reduction::natural_limb> m_literals, m_values;
    device_allocation<reduction::arithmetic_frame> m_frames;
    device_allocation<reduction::natural_column> m_columns;
    device_allocation<reduction::machine> m_states;
    device_allocation<reduction::outcome> m_results;
    device_allocation<std::uint64_t> m_products;
    device_allocation<live_layout> m_live_layout;
    device_allocation<reduction::closure> m_live_arguments;
    device_allocation<reduction::binding> m_live_bindings;
    device_allocation<reduction::natural_limb> m_live_values;
    device_allocation<reduction::arithmetic_frame> m_live_frames;
    explicit resident_storage(sycl::queue & q):m_layout(q), m_code(q), m_arguments(q),
        m_bindings(q), m_literals(q), m_values(q), m_frames(q), m_columns(q),
        m_states(q), m_results(q), m_products(q), m_live_layout(q), m_live_arguments(q),
        m_live_bindings(q), m_live_values(q), m_live_frames(q) {}
    device_batch view() const {
        return {m_layout.data(), m_code.data(), m_arguments.data(), m_bindings.data(),
                m_literals.data(), m_values.data(), m_frames.data(), m_columns.data(),
                m_states.data(), m_results.data(), m_products.data()};
    }
    device_transfer transfer() const {
        return {m_live_layout.data(), m_live_arguments.data(), m_live_bindings.data(),
                m_live_values.data(), m_live_frames.data()};
    }
    void reserve_live(std::size_t regions, std::size_t arguments, std::size_t bindings,
                      std::size_t values, std::size_t frames) {
        m_live_layout.reserve(regions); m_live_arguments.reserve(arguments); m_live_bindings.reserve(bindings);
        m_live_values.reserve(values); m_live_frames.reserve(frames);
    }
    std::size_t bytes() const {
        return m_layout.bytes() + m_code.bytes() + m_arguments.bytes() + m_bindings.bytes() +
            m_literals.bytes() + m_values.bytes() + m_frames.bytes() + m_columns.bytes() +
            m_states.bytes() + m_results.bytes() + m_products.bytes() + m_live_layout.bytes() +
            m_live_arguments.bytes() + m_live_bindings.bytes() + m_live_values.bytes() + m_live_frames.bytes();
    }
    void clear() {
        m_layout.clear(); m_code.clear(); m_arguments.clear(); m_bindings.clear();
        m_literals.clear(); m_values.clear(); m_frames.clear(); m_columns.clear();
        m_states.clear(); m_results.clear(); m_products.clear();
        m_live_layout.clear(); m_live_arguments.clear(); m_live_bindings.clear();
        m_live_values.clear(); m_live_frames.clear();
    }
};

/** Declared after host transfer arrays, so they outlive the drain even when
    enqueueing or execution throws. Normal completion disarms the guard. */
struct submission_drain {
    sycl::queue & m_queue;
    bool m_active = true;
    ~submission_drain() {
        if (m_active) { try { m_queue.wait_and_throw(); } catch (...) {} }
    }
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

template<typename T> buffer_slice reserve_region_slice(std::size_t & total, std::size_t size) {
    if (size > SIZE_MAX / sizeof(T) - total) throw std::length_error("resident region size overflow");
    buffer_slice result{total, size};
    total += size;
    return result;
}

template<typename T>
buffer_slice append_live_prefix(std::vector<T> & packed, std::vector<T> const & region, std::size_t count) {
    auto total = packed.size();
    auto slice = reserve_region_slice<T>(total, count);
    packed.insert(packed.end(), region.begin(), region.begin() + count);
    return slice;
}
}

class reduction_sycl_backend::imp {
    sycl::queue m_queue;
    reduction_sycl_execution m_execution;
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> m_bundle;
    std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> m_transfer_bundle;
    std::size_t m_group_size = 0;
    std::size_t m_transfer_group_size = 0;
    std::uint64_t m_last_products = 0;
    std::size_t m_last_regions = 0;
    bool m_profiling;
    std::optional<std::uint64_t> m_last_kernel_ns;
    bool m_use_resident = false;
    std::unique_ptr<resident_storage> m_resident;
    std::size_t m_last_upload = 0, m_last_download = 0;

    template<typename Kernel>
    void prepare_kernel(std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> & cache,
                        std::size_t & width) {
        if (cache) return;
        auto device = m_queue.get_device();
        auto id = sycl::get_kernel_id<Kernel>();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            m_queue.get_context(), {device}, {id});
        auto maximum = bundle.get_kernel(id).template get_info<sycl::info::kernel_device_specific::work_group_size>(device);
        auto dimension = device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
        auto size = std::min(maximum, dimension);
        if (size == 0) throw reduction_backend_error("device reported an empty work-group capacity");
        cache.reset(new sycl::kernel_bundle<sycl::bundle_state::executable>(std::move(bundle)));
        width = size;
    }
    void prepare_group() {
        if (m_use_resident) prepare_kernel<reduction_resident_cooperative_kernel>(m_bundle, m_group_size);
        else prepare_kernel<reduction_cooperative_kernel>(m_bundle, m_group_size);
    }
    sycl::event transfer_live(resident_storage const & device, std::size_t regions, bool gather,
                              std::vector<sycl::event> const & dependencies) {
        auto width = m_transfer_group_size;
        if (regions > SIZE_MAX / width) throw std::length_error("reduction transfer work range overflow");
        auto live = device.transfer(); auto arena = device.view();
        return m_queue.submit([&](sycl::handler & h) {
            h.depends_on(dependencies);
            h.use_kernel_bundle(*m_transfer_bundle);
            h.parallel_for<reduction_transfer_kernel>(
                sycl::nd_range<1>(sycl::range<1>(regions * width), sycl::range<1>(width)),
                [=](sycl::nd_item<1> item) { transfer_region(live, arena, gather, item); });
        });
    }
public:
    explicit imp(reduction_sycl_execution execution, bool profiling, reduction_sycl_storage storage):
        m_queue(sycl::gpu_selector_v, [](sycl::exception_list errors) {
            for (auto const & error : errors) std::rethrow_exception(error);
        }, profiling ? sycl::property_list{sycl::property::queue::enable_profiling{}} : sycl::property_list{}),
        m_execution(execution), m_profiling(profiling) {
        auto supported = m_queue.get_device().has(sycl::aspect::usm_device_allocations);
        if (storage == reduction_sycl_storage::resident && !supported)
            throw reduction_backend_error("selected device does not support resident allocations");
        m_use_resident = storage == reduction_sycl_storage::resident ||
            (storage == reduction_sycl_storage::automatic && supported);
    }

    std::size_t work_group_size() const {
        return m_execution == reduction_sycl_execution::scalar ? 1 : m_group_size;
    }
    std::uint64_t last_cooperative_products() const { return m_last_products; }
    std::size_t last_region_count() const { return m_last_regions; }
    std::optional<std::uint64_t> last_kernel_nanoseconds() const { return m_last_kernel_ns; }
    bool uses_resident_storage() const { return m_use_resident; }
    std::size_t retained_device_bytes() const { return m_resident ? m_resident->bytes() : 0; }
    std::size_t last_explicit_upload_bytes() const { return m_last_upload; }
    std::size_t last_explicit_download_bytes() const { return m_last_download; }
    void release_storage() { if (m_resident) m_resident->clear(); }

    std::vector<reduction::outcome> submit(std::vector<reduction::submission> & requests) {
        m_last_products = 0; m_last_regions = 0; m_last_kernel_ns.reset();
        m_last_upload = 0; m_last_download = 0;
        if (requests.empty()) return {};
        return m_use_resident ? submit_resident(requests) : submit_buffers(requests);
    }

    std::vector<reduction::outcome> submit_resident(std::vector<reduction::submission> & requests) {
        using namespace reduction;
        bool cooperative = m_execution == reduction_sycl_execution::cooperative;
        if (cooperative) prepare_group();
        prepare_kernel<reduction_transfer_kernel>(m_transfer_bundle, m_transfer_group_size);
        auto group_size = work_group_size();
        if (requests.size() > SIZE_MAX / group_size)
            throw std::length_error("reduction batch work range overflow");
        if (!m_resident) m_resident.reset(new resident_storage(m_queue));
        auto & device = *m_resident;
        std::vector<region_layout> layout;
        std::vector<instruction> code;
        std::vector<natural_limb> literals;
        std::vector<machine> next;
        std::vector<live_layout> live;
        std::vector<closure> live_arguments;
        std::vector<binding> live_bindings;
        std::vector<natural_limb> live_values;
        std::vector<arithmetic_frame> live_frames;
        std::vector<outcome> results(requests.size(), outcome::invalid);
        std::vector<std::uint64_t> products(requests.size(), 0);
        std::size_t code_count = 0, literal_count = 0;
        std::size_t argument_count = 0, binding_count = 0, value_count = 0, frame_count = 0, column_count = 0;
        for (auto const & r : requests) {
            reserve_region_slice<instruction>(code_count, r.m_code.size());
            reserve_region_slice<natural_limb>(literal_count, r.m_arithmetic.m_literals.size());
        }
        code.reserve(code_count); literals.reserve(literal_count);
        layout.reserve(requests.size()); next.reserve(requests.size());
        for (auto const & r : requests) {
            layout.push_back({append_region_buffer(code, r.m_code),
                reserve_region_slice<closure>(argument_count, r.m_arguments.size()),
                reserve_region_slice<binding>(binding_count, r.m_bindings.size()),
                append_region_buffer(literals, r.m_arithmetic.m_literals),
                reserve_region_slice<natural_limb>(value_count, r.m_arithmetic.m_values.size()),
                reserve_region_slice<arithmetic_frame>(frame_count, r.m_arithmetic.m_frames.size()),
                reserve_region_slice<natural_column>(column_count, cooperative ? r.m_arithmetic.m_columns : 0)});
            next.push_back(r.m_state);
            auto const & state = r.m_state;
            bool valid = state.m_num_arguments <= r.m_arguments.size() &&
                state.m_num_bindings <= r.m_bindings.size() &&
                state.m_num_values <= r.m_arithmetic.m_values.size() &&
                state.m_num_frames <= r.m_arithmetic.m_frames.size();
            // Bad counters cannot become host reads; the machine rejects them.
            live.push_back({append_live_prefix(live_arguments, r.m_arguments, valid ? state.m_num_arguments : 0),
                append_live_prefix(live_bindings, r.m_bindings, valid ? state.m_num_bindings : 0),
                append_live_prefix(live_values, r.m_arithmetic.m_values, valid ? state.m_num_values : 0),
                append_live_prefix(live_frames, r.m_arithmetic.m_frames, valid ? state.m_num_frames : 0)});
        }
        // Idle allocations can be replaced without losing a checkpoint. No
        // previous device contents are used as evidence that inputs match.
        device.m_layout.reserve(layout.size()); device.m_code.reserve(code.size());
        device.m_literals.reserve(literals.size()); device.m_states.reserve(next.size());
        device.m_results.reserve(results.size()); device.m_products.reserve(products.size());
        device.m_arguments.reserve(argument_count); device.m_bindings.reserve(binding_count);
        device.m_values.reserve(value_count); device.m_frames.reserve(frame_count);
        device.m_columns.reserve(column_count);
        device.reserve_live(live.size(), live_arguments.size(), live_bindings.size(), live_values.size(), live_frames.size());
        std::vector<sycl::event> uploaded;
        std::size_t upload_bytes = 0, download_bytes = 0;
        submission_drain drain{m_queue};
        auto upload = [&]<typename T>(T * destination, T const * source, std::size_t count) {
            if (count == 0) return;
            uploaded.push_back(m_queue.memcpy(destination, source, count * sizeof(T)));
            upload_bytes += count * sizeof(T);
        };
        upload(device.m_layout.data(), layout.data(), layout.size());
        upload(device.m_code.data(), code.data(), code.size());
        upload(device.m_literals.data(), literals.data(), literals.size());
        upload(device.m_states.data(), next.data(), next.size());
        upload(device.m_results.data(), results.data(), results.size());
        upload(device.m_products.data(), products.data(), products.size());
        upload(device.m_live_layout.data(), live.data(), live.size());
        upload(device.m_live_arguments.data(), live_arguments.data(), live_arguments.size());
        upload(device.m_live_bindings.data(), live_bindings.data(), live_bindings.size());
        upload(device.m_live_values.data(), live_values.data(), live_values.size());
        upload(device.m_live_frames.data(), live_frames.data(), live_frames.size());
        auto unpacked = transfer_live(device, requests.size(), false, uploaded);
        auto view = device.view();
        auto completion = m_queue.submit([&](sycl::handler & h) {
            h.depends_on(unpacked);
            if (!cooperative) {
                h.parallel_for<reduction_resident_scalar_kernel>(sycl::range<1>(requests.size()),
                    [=](sycl::id<1> item) { run_scalar(view, item[0]); });
            } else {
                h.use_kernel_bundle(*m_bundle);
                h.parallel_for<reduction_resident_cooperative_kernel>(
                    sycl::nd_range<1>(sycl::range<1>(requests.size() * group_size), sycl::range<1>(group_size)),
                    [=](sycl::nd_item<1> item) { run_cooperative(view, item); });
            }
        });
        auto ready = completion;
        auto download = [&]<typename T>(T * destination, T const * source, std::size_t count) {
            if (count == 0) return;
            m_queue.memcpy(destination, source, count * sizeof(T), ready);
            download_bytes += count * sizeof(T);
        };
        download(next.data(), device.m_states.data(), next.size());
        download(results.data(), device.m_results.data(), results.size());
        download(products.data(), device.m_products.data(), products.size());
        m_queue.wait_and_throw();
        // Counter round trip determines compact output offsets; a single
        // gather precedes the bulk downloads. Host checkpoint ownership does
        // not change until the enclosing session validates every outcome.
        live.clear();
        std::size_t live_argument_count = 0, live_binding_count = 0, live_value_count = 0, live_frame_count = 0;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto const & s = layout[i]; auto const & state = next[i];
            if (results[i] == outcome::invalid || state.m_num_arguments > s.m_arguments.m_size ||
                state.m_num_bindings > s.m_bindings.m_size || state.m_num_values > s.m_values.m_size ||
                state.m_num_frames > s.m_frames.m_size) {
                results[i] = outcome::invalid;
            }
            bool valid = results[i] != outcome::invalid;
            live.push_back({reserve_region_slice<closure>(live_argument_count, valid ? state.m_num_arguments : 0),
                reserve_region_slice<binding>(live_binding_count, valid ? state.m_num_bindings : 0),
                reserve_region_slice<natural_limb>(live_value_count, valid ? state.m_num_values : 0),
                reserve_region_slice<arithmetic_frame>(live_frame_count, valid ? state.m_num_frames : 0)});
        }
        live_arguments.resize(live_argument_count); live_bindings.resize(live_binding_count);
        live_values.resize(live_value_count); live_frames.resize(live_frame_count);
        device.reserve_live(live.size(), live_argument_count, live_binding_count, live_value_count, live_frame_count);
        uploaded.clear();
        upload(device.m_live_layout.data(), live.data(), live.size());
        ready = transfer_live(device, requests.size(), true, uploaded);
        download(live_arguments.data(), device.m_live_arguments.data(), live_arguments.size());
        download(live_bindings.data(), device.m_live_bindings.data(), live_bindings.size());
        download(live_values.data(), device.m_live_values.data(), live_values.size());
        download(live_frames.data(), device.m_live_frames.data(), live_frames.size());
        m_queue.wait_and_throw();
        std::optional<std::uint64_t> kernel_ns;
        if (m_profiling) {
            auto start = completion.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto finish = completion.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (finish < start) throw reduction_backend_error("device returned reversed profiling timestamps");
            kernel_ns = finish - start;
        }
        m_queue.throw_asynchronous();
        drain.m_active = false;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto & r = requests[i]; auto const & p = live[i];
            restore_region_buffer(live_arguments, p.m_arguments, r.m_arguments);
            restore_region_buffer(live_bindings, p.m_bindings, r.m_bindings);
            restore_region_buffer(live_values, p.m_values, r.m_arithmetic.m_values);
            restore_region_buffer(live_frames, p.m_frames, r.m_arithmetic.m_frames);
            requests[i].m_state = next[i];
            m_last_products += std::min(products[i], UINT64_MAX - m_last_products);
        }
        m_last_regions = requests.size(); m_last_kernel_ns = kernel_ns;
        m_last_upload = upload_bytes; m_last_download = download_bytes;
        return results;
    }

    std::vector<reduction::outcome> submit_buffers(std::vector<reduction::submission> & requests) {
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
                auto view = [=](natural_column * scratch) {
                    return device_batch{slices.get_multi_ptr<sycl::access::decorated::no>().get(),
                        instructions.get_multi_ptr<sycl::access::decorated::no>().get(),
                        a.get_multi_ptr<sycl::access::decorated::no>().get(),
                        b.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l.get_multi_ptr<sycl::access::decorated::no>().get(),
                        v.get_multi_ptr<sycl::access::decorated::no>().get(),
                        f.get_multi_ptr<sycl::access::decorated::no>().get(), scratch,
                        m.get_multi_ptr<sycl::access::decorated::no>().get(),
                        out.get_multi_ptr<sycl::access::decorated::no>().get(),
                        completed_products.get_multi_ptr<sycl::access::decorated::no>().get()};
                };
                if (!cooperative) {
                    h.parallel_for<reduction_scalar_kernel>(sycl::range<1>(requests.size()), [=](sycl::id<1> item) {
                        run_scalar(view(nullptr), item[0]);
                    });
                } else {
                    h.use_kernel_bundle(*m_bundle);
                    auto c = columns->get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for<reduction_cooperative_kernel>(
                        sycl::nd_range<1>(sycl::range<1>(requests.size() * group_size), sycl::range<1>(group_size)),
                        [=](sycl::nd_item<1> item) {
                            run_cooperative(view(c.get_multi_ptr<sycl::access::decorated::no>().get()), item);
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

reduction_sycl_backend::reduction_sycl_backend(reduction_sycl_execution execution, bool profiling,
                                               reduction_sycl_storage storage):
    m_execution(execution), m_profiling(profiling), m_storage(storage) {}
reduction_sycl_backend::~reduction_sycl_backend() = default;

std::size_t reduction_sycl_backend::work_group_size() const { return m_imp ? m_imp->work_group_size() : 0; }
std::uint64_t reduction_sycl_backend::last_cooperative_products() const {
    return m_imp ? m_imp->last_cooperative_products() : 0;
}
std::size_t reduction_sycl_backend::last_region_count() const { return m_imp ? m_imp->last_region_count() : 0; }
std::optional<std::uint64_t> reduction_sycl_backend::last_kernel_nanoseconds() const {
    return m_imp ? m_imp->last_kernel_nanoseconds() : std::nullopt;
}
bool reduction_sycl_backend::uses_resident_storage() const { return m_imp && m_imp->uses_resident_storage(); }
std::size_t reduction_sycl_backend::retained_device_bytes() const { return m_imp ? m_imp->retained_device_bytes() : 0; }
std::size_t reduction_sycl_backend::last_explicit_upload_bytes() const {
    return m_imp ? m_imp->last_explicit_upload_bytes() : 0;
}
std::size_t reduction_sycl_backend::last_explicit_download_bytes() const {
    return m_imp ? m_imp->last_explicit_download_bytes() : 0;
}
void reduction_sycl_backend::release_storage() {
    try {
        if (m_imp) m_imp->release_storage();
    } catch (sycl::exception const & error) {
        m_imp.reset();
        throw reduction_backend_error(error.what());
    } catch (...) {
        m_imp.reset();
        throw;
    }
}

std::vector<reduction::outcome> reduction_sycl_backend::submit(std::vector<reduction::submission> & requests) {
    try {
        // Empty batches need no GPU, including when no platform is available.
        if (!m_imp && requests.empty()) return {};
        if (!m_imp) m_imp.reset(new imp(m_execution, m_profiling, m_storage));
        return m_imp->submit(requests);
    } catch (sycl::exception const & error) {
        m_imp.reset();
        throw reduction_backend_error(error.what());
    } catch (...) {
        // No cached device workspace is reused after a failed submission.
        // The method-local drain ran while all borrowed host data was alive.
        m_imp.reset();
        throw;
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
