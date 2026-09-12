/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace lean {
namespace reduction {

/** Canonical, non-owning natural numbers for portable reduction backends.
    Limbs are little endian in base 2^32. Zero has no limbs; a nonzero value
    has a nonzero last limb. There is no fixed precision or truncation.

    This representation never contains Lean objects, allocator state or host
    pointers embedded in device data. A backend supplies views of its own
    storage. Input views remain valid for the duration of an operation. */
using natural_limb = std::uint32_t;
struct natural_view {
    natural_limb const * m_data;
    std::size_t m_size;
};

enum class natural_status { complete, need_space, invalid };
struct natural_result {
    natural_status m_status;
    // Complete: canonical result length. Need_space: required capacity.
    // Invalid: zero. Unused destination limbs do not form part of the value.
    std::size_t m_size;
};

inline bool valid_natural(natural_view a) {
    return a.m_size == 0 || (a.m_data != nullptr && a.m_data[a.m_size - 1] != 0);
}

inline int compare_natural(natural_view a, natural_view b) {
    if (a.m_size != b.m_size) return a.m_size < b.m_size ? -1 : 1;
    for (std::size_t i = a.m_size; i != 0; --i) {
        if (a.m_data[i - 1] != b.m_data[i - 1])
            return a.m_data[i - 1] < b.m_data[i - 1] ? -1 : 1;
    }
    return 0;
}

inline std::size_t natural_size(natural_limb const * data, std::size_t size) {
    while (size != 0 && data[size - 1] == 0) --size;
    return size;
}

/** Destinations must be disjoint from both inputs. Aliasing is a caller
    ownership violation, not a supported in-place mode. Every operation checks
    shape and capacity before writing, so need_space leaves all storage intact.
    Growing the destination and retrying is a resource action, not a change to
    the mathematical result. Arithmetic has no step/precision/depth ceiling. */
inline natural_result add_natural(natural_view a, natural_view b,
                                  natural_limb * out, std::size_t capacity) {
    if (!valid_natural(a) || !valid_natural(b)) return {natural_status::invalid, 0};
    std::size_t size = a.m_size > b.m_size ? a.m_size : b.m_size;
    if (size == 0) return {natural_status::complete, 0};
    if (size == std::numeric_limits<std::size_t>::max()) return {natural_status::invalid, 0};
    std::size_t needed = size + 1;
    if (capacity < needed) return {natural_status::need_space, needed};
    if (out == nullptr) return {natural_status::invalid, 0};
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < size; ++i) {
        std::uint64_t sum = carry;
        if (i < a.m_size) sum += a.m_data[i];
        if (i < b.m_size) sum += b.m_data[i];
        out[i] = static_cast<natural_limb>(sum);
        carry = sum >> 32;
    }
    out[size] = static_cast<natural_limb>(carry);
    return {natural_status::complete, size + (carry != 0)};
}

/** Natural-number subtraction saturates at zero, as in Lean's Nat.sub. */
inline natural_result subtract_natural(natural_view a, natural_view b,
                                       natural_limb * out, std::size_t capacity) {
    if (!valid_natural(a) || !valid_natural(b)) return {natural_status::invalid, 0};
    if (compare_natural(a, b) <= 0) return {natural_status::complete, 0};
    if (capacity < a.m_size) return {natural_status::need_space, a.m_size};
    if (out == nullptr) return {natural_status::invalid, 0};
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < a.m_size; ++i) {
        std::uint64_t sub = borrow + (i < b.m_size ? b.m_data[i] : 0);
        std::uint64_t value = a.m_data[i];
        out[i] = static_cast<natural_limb>(value - sub);
        borrow = value < sub;
    }
    return {natural_status::complete, natural_size(out, a.m_size)};
}

/** Exact schoolbook multiplication. Each accumulator is at most
    (2^32-1)^2 + (2^32-1) + (2^32-1) = 2^64-1, so no term or carry is lost.
    Independent invocations can run in parallel without shared mutable state.
    This is the portable baseline, not a claim of an optimal large multiply. */
inline natural_result multiply_natural(natural_view a, natural_view b,
                                       natural_limb * out, std::size_t capacity) {
    if (!valid_natural(a) || !valid_natural(b)) return {natural_status::invalid, 0};
    if (a.m_size == 0 || b.m_size == 0) return {natural_status::complete, 0};
    if (a.m_size > std::numeric_limits<std::size_t>::max() - b.m_size)
        return {natural_status::invalid, 0};
    std::size_t size = a.m_size + b.m_size;
    if (capacity < size) return {natural_status::need_space, size};
    if (out == nullptr) return {natural_status::invalid, 0};
    for (std::size_t i = 0; i < size; ++i) out[i] = 0;
    for (std::size_t i = 0; i < a.m_size; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < b.m_size; ++j) {
            std::uint64_t product = static_cast<std::uint64_t>(a.m_data[i]) * b.m_data[j];
            product += out[i + j];
            product += carry;
            out[i + j] = static_cast<natural_limb>(product);
            carry = product >> 32;
        }
        // Earlier rows end before this cell; this is not an accumulating store.
        out[i + b.m_size] = static_cast<natural_limb>(carry);
    }
    return {natural_status::complete, natural_size(out, size)};
}

/** Independent convolution columns for cooperative or batched multiplication.
    A column contains at most min(a.size,b.size) products of 32-bit limbs.
    Two 64-bit words hold their exact sum even with 64-bit size_t; uint64_t
    alone would overflow after only two maximal products. Column evaluation
    has no cross-thread mutation or synchronization requirement. */
struct natural_column {
    std::uint64_t m_low = 0;
    std::uint64_t m_high = 0;

    void add(std::uint64_t value) {
        auto before = m_low;
        m_low += value;
        m_high += m_low < before;
    }
};

/** The caller validates canonical inputs and representable a.size+b.size
    once before dispatch, and provides 0 <= column < a.size+b.size. */
inline natural_column multiply_natural_column(natural_view a, natural_view b, std::size_t column) {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    natural_column result;
    if (a.m_size == 0 || b.m_size == 0) return result;
    std::size_t first = column >= b.m_size ? column - (b.m_size - 1) : 0;
    std::size_t last = column < a.m_size ? column : a.m_size - 1;
    for (std::size_t i = first; i <= last; ++i)
        result.add(static_cast<std::uint64_t>(a.m_data[i]) * b.m_data[column - i]);
    return result;
}

/** Carry propagation is linear in the output size, following independent
    quadratic-work columns. The temporary column array and output are disjoint.
    There must be exactly a.size+b.size columns for the nonzero product,
    including its final zero convolution column. Zero uses no columns.

    Exact column sums plus incoming carry fit two words: with n products,
    their combined value is bounded by n*(2^32-1)*2^32 < 2^128. A final
    residual carry is an invalid column stream, never a truncated result. */
inline natural_result finish_natural_product(natural_column const * columns, std::size_t size,
                                             natural_limb * out, std::size_t capacity) {
    if (capacity < size) return {natural_status::need_space, size};
    if (size == 0) return {natural_status::complete, 0};
    if (columns == nullptr || out == nullptr) return {natural_status::invalid, 0};
    natural_column carry;
    for (std::size_t i = 0; i < size; ++i) {
        auto sum = columns[i];
        auto high = sum.m_high;
        sum.add(carry.m_low);
        if (sum.m_high < high) return {natural_status::invalid, 0};
        high = sum.m_high;
        sum.m_high += carry.m_high;
        if (sum.m_high < high) return {natural_status::invalid, 0};
        out[i] = static_cast<natural_limb>(sum.m_low);
        carry = {(sum.m_low >> 32) | (sum.m_high << 32), sum.m_high >> 32};
    }
    if (carry.m_low != 0 || carry.m_high != 0) return {natural_status::invalid, 0};
    return {natural_status::complete, natural_size(out, size)};
}

}
}
