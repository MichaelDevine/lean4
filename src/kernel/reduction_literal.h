/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <stdexcept>
#include <vector>
#include "kernel/expr.h"
#include "kernel/reduction_machine.h"

namespace lean {

#ifdef LEAN_USE_GMP
class reduction_mpz_buffer {
    mpz_t m_value;
public:
    reduction_mpz_buffer() { mpz_init(m_value); }
    ~reduction_mpz_buffer() { mpz_clear(m_value); }
    reduction_mpz_buffer(reduction_mpz_buffer const &) = delete;
    reduction_mpz_buffer & operator=(reduction_mpz_buffer const &) = delete;
    mpz_ptr data() { return m_value; }
};
#endif

/** Copy a literal's exact limbs; never normalize its enclosing expression.
    GMP export/import are linear and avoid decimal text or repeated large
    divisions on the normal GMP build. The portable runtime stays supported. */
inline reduction::natural_reference append_reduction_literal(
    nat const & value, std::vector<reduction::natural_limb> & limbs) {
    auto offset = limbs.size();
    if (value.is_zero()) return {offset, 0, false};
#ifdef LEAN_USE_GMP
    reduction_mpz_buffer raw;
    value.to_mpz().set(raw.data());
    auto bits = mpz_sizeinbase(raw.data(), 2);
    auto count = bits / 32 + (bits % 32 != 0);
    if (count > limbs.max_size() - offset) throw std::length_error("literal limb capacity overflow");
    limbs.resize(offset + count);
    std::size_t written = 0;
    mpz_export(limbs.data() + offset, &written, -1, sizeof(reduction::natural_limb), 0, 0, raw.data());
    if (written != count) throw std::logic_error("literal export length mismatch");
#else
    nat remaining = value;
    nat base("4294967296");
    while (!remaining.is_zero()) {
        limbs.push_back((remaining % base).to_mpz().get_unsigned_int());
        remaining = remaining / base;
    }
#endif
    return {offset, limbs.size() - offset, false};
}

inline expr reify_reduction_natural(reduction::natural_view value) {
    if (!reduction::valid_natural(value)) throw std::invalid_argument("invalid numeric reduction result");
    if (value.m_size == 0) return mk_lit(literal(0u));
#ifdef LEAN_USE_GMP
    reduction_mpz_buffer raw;
    mpz_import(raw.data(), value.m_size, -1, sizeof(reduction::natural_limb), 0, 0, value.m_data);
    return mk_lit(literal(nat(mpz(raw.data()))));
#else
    nat result, base("4294967296");
    for (auto i = value.m_size; i != 0; --i) result = result * base + nat(value.m_data[i - 1]);
    return mk_lit(literal(result));
#endif
}
}
