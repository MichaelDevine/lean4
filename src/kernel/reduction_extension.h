/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include "kernel/reduction_region.h"

namespace lean {

/** Optional trusted implementation of kernel reduction, not a tactic or
    certificate oracle. A result must be definitionally equal to the input
    and in the weak head normal form prescribed by the exact request mode.
    The kernel does not re-evaluate it to independently verify that contract.
    Implementations belong to the trusted computing base when enabled.

    Return none for unsupported, unprofitable or unavailable work. Such a
    decline leaves the original expression for the ordinary CPU reducer.
    Device/resource failure must not return a successful expression. */
class reduction_extension {
public:
    virtual ~reduction_extension() = default;
    virtual optional<expr> reduce(environment const &, local_ctx const &,
                                  expr const &, reduction_mode) = 0;
};

/** Install a borrowed extension on this thread for a lexical scope. The
    extension must outlive the scope. Nested scopes restore the prior value;
    parallel threads must opt in independently. No production default changes. */
class scope_reduction_extension {
    reduction_extension * m_previous;
public:
    explicit scope_reduction_extension(reduction_extension *);
    ~scope_reduction_extension();
    scope_reduction_extension(scope_reduction_extension const &) = delete;
    scope_reduction_extension & operator=(scope_reduction_extension const &) = delete;
};

optional<expr> try_reduction_extension(environment const &, local_ctx const &,
                                      expr const &, reduction_mode);
}
