/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <stdexcept>
#include <utility>
#include <vector>
#include "util/alloc.h"
#include "kernel/expr_cache_hash.h"

namespace lean {
/* An insert-only structural expression cache for one type-checker state.

   Both handles in an unused slot hold box(0), the ordinary empty/moved-from
   object_ref representation, NOT expr() (which is a valid dummy constant).
   These private empty handles are only moved/destroyed; they are never used
   as expressions. All actual Expr constructors have non-scalar objects.

   No eviction, key approximation, or reduction rule is introduced. Lookup
   returns a borrowed result invalidated by insertion; callers must copy it
   before any recursive evaluation. */
class expr_whnf_cache {
    struct entry {
        expr   m_key;
        expr   m_value;
        size_t m_hash = 0;

        entry():m_key(box(0)), m_value(box(0)) {}
        bool empty() const { return is_scalar(m_key.raw()); }
    };
    typedef std::vector<entry, lean::allocator<entry>> entries;
    entries m_entries;
    size_t  m_size = 0;

    size_t position(expr const & key, size_t hash) const {
        size_t mask = m_entries.size() - 1;
        size_t i = hash & mask;
        while (!m_entries[i].empty()) {
            auto const & slot = m_entries[i];
            if (slot.m_hash == hash && slot.m_key == key)
                break;
            i = (i + 1) & mask;
        }
        return i;
    }

    void grow() {
        size_t old_capacity = m_entries.size();
        // Allocate first. Allocation/length failure leaves every owned entry
        // intact; moving the handles afterward neither allocates nor throws.
        if (old_capacity > m_entries.max_size() / 2)
            throw std::length_error("expression cache capacity overflow");
        size_t capacity = old_capacity == 0 ? 16 : 2 * old_capacity;
        entries next(capacity);
        for (auto & slot : m_entries) {
            if (slot.empty())
                continue;
            size_t i = slot.m_hash & (capacity - 1);
            while (!next[i].empty())
                i = (i + 1) & (capacity - 1);
            next[i].m_key = std::move(slot.m_key);
            next[i].m_value = std::move(slot.m_value);
            next[i].m_hash = slot.m_hash;
        }
        m_entries.swap(next);
    }

public:
    expr_whnf_cache() = default;
    expr_whnf_cache(expr_whnf_cache const &) = default;
    expr_whnf_cache(expr_whnf_cache && other) noexcept {
        swap(other);
    }
    expr_whnf_cache & operator=(expr_whnf_cache other) {
        swap(other);
        return *this;
    }

    void swap(expr_whnf_cache & other) noexcept {
        m_entries.swap(other.m_entries);
        std::swap(m_size, other.m_size);
    }

    size_t size() const { return m_size; }

    expr const * find(expr const & key) const {
        if (m_entries.empty())
            return nullptr;
        auto const & slot = m_entries[position(key, expr_cache_hash()(key))];
        return slot.empty() ? nullptr : &slot.m_value;
    }

    /* First insertion wins, including when recursive checking inserted the
       key before an outer evaluation returned. Own both arguments before a
       possible growth, so aliases of existing results are also safe. */
    void insert(expr const & key, expr const & value) {
        size_t hash = expr_cache_hash()(key);
        if (!m_entries.empty() && !m_entries[position(key, hash)].empty())
            return;
        expr owned_key(key), owned_value(value);
        // A physical table load factor, not a limit on checking or cache size.
        // Retain an empty probe slot and avoid overflow in the load test.
        if (m_entries.empty() || m_size >= m_entries.size() - m_entries.size() / 4)
            grow();
        size_t i = hash & (m_entries.size() - 1);
        while (!m_entries[i].empty())
            i = (i + 1) & (m_entries.size() - 1);
        auto & slot = m_entries[i];
        slot.m_key = std::move(owned_key);
        slot.m_value = std::move(owned_value);
        slot.m_hash = hash;
        ++m_size;
    }
};
}
