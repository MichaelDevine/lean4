/* Test-only linker interception; no runtime or cache implementation changes. */
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>
#include "runtime/init_module.h"
#include "kernel/expr_whnf_cache.h"

#ifndef LEAN_MIMALLOC
#error "Allocation-failure control requires the actual LEAN_MIMALLOC allocator"
#endif

extern "C" void lean_initialize();
extern "C" void * __real_mi_new_n(size_t count, size_t size);
using namespace lean;

static unsigned checks = 0;
static unsigned retries = 0;
static thread_local bool armed = false;
static thread_local size_t expected_count = 0;
static thread_local unsigned injected = 0;
static thread_local unsigned forwarded = 0;
static thread_local size_t last_count = 0;
static thread_local size_t last_size = 0;

static void require(bool condition, char const * message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "check failed: %s\n", message);
        std::exit(1);
    }
}

extern "C" void * __wrap_mi_new_n(size_t count, size_t size) {
    if (armed) {
        armed = false;
        // The reviewed private slot consists of two expr handles and a hash.
        // Reject interception of an unrelated allocation instead of treating
        // a different failure site as evidence about the cache array.
        require(count == expected_count && size == 2 * sizeof(expr) + sizeof(size_t),
                "armed allocation is not the expected cache array");
        ++injected;
        throw std::bad_alloc();
    }
    ++forwarded;
    last_count = count;
    last_size = size;
    return __real_mi_new_n(count, size);
}

template<class F> static void fail_once(size_t count, F operation) {
    require(!armed, "nested allocation-failure arm");
    auto before = injected;
    expected_count = count;
    armed = true;
    bool caught = false;
    try {
        operation();
    } catch (std::bad_alloc const &) {
        caught = true;
    } catch (...) {
        armed = false;
        require(false, "operation threw an exception other than bad_alloc");
    }
    bool consumed = !armed;
    armed = false;
    require(caught && consumed && injected == before + 1,
            "operation did not consume exactly one injected bad_alloc");
}

template<class F> static void retry(size_t count, F operation) {
    require(!armed, "retry still armed");
    auto before = forwarded;
    operation();
    require(forwarded == before + 1 && last_count == count &&
            last_size == 2 * sizeof(expr) + sizeof(size_t),
            "retry did not allocate the expected array through the real allocator");
    ++retries;
}

struct inputs {
    std::vector<expr> keys, values;
    std::vector<int> key_rc, value_rc;

    explicit inputs(unsigned count, unsigned offset = 0) {
        expr key_fn = mk_const(name("allocationKey"));
        expr value_fn = mk_const(name("allocationValue"));
        for (unsigned i = 0; i < count; ++i) {
            keys.push_back(mk_app(key_fn, mk_lit(literal(offset + i))));
            values.push_back(mk_app(value_fn, mk_lit(literal(offset + i))));
        }
        for (unsigned i = 0; i < count; ++i) {
            key_rc.push_back(keys[i].raw()->m_rc);
            value_rc.push_back(values[i].raw()->m_rc);
            require(key_rc.back() > 0 && value_rc.back() > 0,
                    "ownership fixture is persistent or shared-threaded");
        }
    }

    void fill(expr_whnf_cache & cache, unsigned count) const {
        for (unsigned i = 0; i < count; ++i) cache.insert(keys[i], values[i]);
    }

    void contents(expr_whnf_cache const & cache, unsigned count) const {
        for (unsigned i = 0; i < count; ++i) {
            auto value = cache.find(keys[i]);
            require(value && *value == values[i], "existing entry changed after allocation failure or retry");
        }
    }

    void references(unsigned owners, unsigned count, unsigned first_value_extra = 0) const {
        for (unsigned i = 0; i < keys.size(); ++i) {
            int held = i < count ? static_cast<int>(owners) : 0;
            int extra = i == 0 ? static_cast<int>(first_value_extra) : 0;
            require(keys[i].raw()->m_rc == key_rc[i] + held &&
                    values[i].raw()->m_rc == value_rc[i] + held + extra,
                    "failed operation, retry or destruction changed exact ownership");
        }
    }
};

int main() {
    initialize_runtime_module();
    lean_initialize();
    lean_io_mark_end_initialization();

    // Initial allocation must release both speculative incoming handle copies.
    inputs initial(1);
    {
        expr_whnf_cache cache;
        fail_once(16, [&] { cache.insert(initial.keys[0], initial.values[0]); });
        require(cache.size() == 0 && !cache.find(initial.keys[0]), "failed initial allocation changed empty map");
        initial.references(0, 1);
        retry(16, [&] { cache.insert(initial.keys[0], initial.values[0]); });
        require(cache.size() == 1, "initial retry has wrong size");
        initial.contents(cache, 1);
        initial.references(1, 1);
    }
    initial.references(0, 1);

    // Twelve entries occupy the initial 16-slot table at its growth threshold.
    inputs growing(13);
    {
        expr_whnf_cache cache;
        growing.fill(cache, 12);
        auto retained = cache.find(growing.keys[0]);
        fail_once(32, [&] { cache.insert(growing.keys[12], growing.values[12]); });
        require(cache.size() == 12 && !cache.find(growing.keys[12]), "failed growth inserted an entry");
        require(cache.find(growing.keys[0]) == retained, "failed growth replaced existing array");
        growing.contents(cache, 12);
        growing.references(1, 12);
        retry(32, [&] { cache.insert(growing.keys[12], growing.values[12]); });
        require(cache.size() == 13, "growth retry has wrong size");
        growing.contents(cache, 13);
        growing.references(1, 13);
    }
    growing.references(0, 13);

    // Both incoming references alias a value in the array whose growth fails.
    inputs aliases(12);
    {
        expr_whnf_cache cache;
        aliases.fill(cache, 12);
        auto borrowed = cache.find(aliases.keys[0]);
        require(borrowed != nullptr, "alias fixture missing");
        fail_once(32, [&] { cache.insert(*borrowed, *borrowed); });
        require(cache.size() == 12 && !cache.find(aliases.values[0]), "failed aliased insertion changed map");
        require(cache.find(aliases.keys[0]) == borrowed, "failed aliased growth invalidated input pointer");
        aliases.contents(cache, 12);
        aliases.references(1, 12);
        retry(32, [&] { cache.insert(*borrowed, *borrowed); });
        // The old borrowed pointer is invalid after successful growth.
        require(cache.size() == 13, "aliased retry has wrong size");
        auto inserted = cache.find(aliases.values[0]);
        require(inserted && *inserted == aliases.values[0], "aliased retry lost new key or value");
        aliases.contents(cache, 12);
        aliases.references(1, 12, 2);
    }
    aliases.references(0, 12);

    // Failure while constructing the by-value assignment parameter must leave
    // both maps, including the old destination's ownership, unchanged.
    inputs source(12), destination(1, 1000);
    {
        expr_whnf_cache from, to;
        source.fill(from, 12);
        destination.fill(to, 1);
        auto old_from = from.find(source.keys[0]);
        auto old_to = to.find(destination.keys[0]);
        fail_once(16, [&] { to = from; });
        require(from.size() == 12 && to.size() == 1, "failed assignment changed map sizes");
        require(from.find(source.keys[0]) == old_from && to.find(destination.keys[0]) == old_to,
                "failed assignment replaced an existing array");
        source.contents(from, 12);
        destination.contents(to, 1);
        source.references(1, 12);
        destination.references(1, 1);
        retry(16, [&] { to = from; });
        require(from.size() == 12 && to.size() == 12 && !to.find(destination.keys[0]),
                "assignment retry retained old destination entries");
        source.contents(from, 12);
        source.contents(to, 12);
        source.references(2, 12);
        destination.references(0, 1);
    }
    source.references(0, 12);
    destination.references(0, 1);
    require(!armed && injected == 4 && retries == 4, "not all exception cases and real retries ran");
    std::printf("{\"valid\":true,\"checks\":%u,\"injected_failures\":%u,\"real_retries\":%u,\"allocator\":\"mi_new_n\"}\n",
                checks, injected, retries);
}
