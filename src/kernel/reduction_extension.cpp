/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#include "runtime/thread.h"
#include "kernel/reduction_extension.h"

namespace lean {
LEAN_THREAD_PTR(reduction_extension, g_reduction_extension);

scope_reduction_extension::scope_reduction_extension(reduction_extension * extension):
    m_previous(g_reduction_extension) {
    g_reduction_extension = extension;
}

scope_reduction_extension::~scope_reduction_extension() {
    g_reduction_extension = m_previous;
}

optional<expr> try_reduction_extension(environment const & env, local_ctx const & lctx,
                                      expr const & e, reduction_mode mode) {
    auto extension = g_reduction_extension;
    if (!extension) return none_expr();
    // An extension may consult the CPU checker; that must not redispatch to
    // itself, including on exceptions. Other threads are unaffected.
    scope_reduction_extension disable(nullptr);
    return extension->reduce(env, lctx, e, mode);
}
}
