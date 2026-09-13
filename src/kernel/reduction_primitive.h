/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include "kernel/expr.h"

namespace lean {

/** Primitive recognition shared by the ordinary checker and captured regions.
    Exact constants, universe arguments and application arity matter. A region
    must not replace a primitive it cannot execute with the declaration body:
    full whnf tries primitive reduction before delta reduction. */
enum class reduction_primitive {
    none, successor, add, subtract, multiply, power, gcd, modulo, divide,
    equal, less_equal, bit_and, bit_or, bit_xor, shift_left, shift_right,
    native_bool, native_nat
};

reduction_primitive classify_reduction_primitive(expr const & function, unsigned arity);

}
