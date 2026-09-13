/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include "kernel/expr.h"

namespace lean {
/* A partial structural fingerprint for in-memory expression caches.

   Keep the stored expression hash unchanged: it is part of imported objects.
   Inspect at most two child edges from the root, including full natural values
   at the boundary. This reaches both operands of ((f a) b) without walking an
   unbounded expression or retaining a second cache of its subexpressions.

   The fingerprint ignores binder names and annotations, just like operator==.
   It may omit other fields that equality compares (e.g. metadata payloads);
   collisions still require the original exact expression comparison. Its
   value never depends on pointer identity, reference counts or DAG sharing. */
struct expr_cache_hash {
private:
    static uint64 hash_expr(expr const & e, unsigned depth) {
        uint64 r = lean::hash(hash(e), static_cast<unsigned>(e.kind()));
        if (is_nat_lit(e)) {
            nat const & n = lit_value(e).get_nat();
            return lean::hash(r, n.is_small() ? static_cast<uint64>(n.get_small_value()) : n.get_big_value().content_hash());
        }
        if (depth == 0) return r;
        switch (e.kind()) {
        case expr_kind::App:
            return lean::hash(lean::hash(r, hash_expr(app_fn(e), depth - 1)), hash_expr(app_arg(e), depth - 1));
        case expr_kind::Lambda: case expr_kind::Pi:
            return lean::hash(lean::hash(r, hash_expr(binding_domain(e), depth - 1)), hash_expr(binding_body(e), depth - 1));
        case expr_kind::Let:
            r = lean::hash(r, hash_expr(let_type(e), depth - 1));
            r = lean::hash(r, hash_expr(let_value(e), depth - 1));
            return lean::hash(r, hash_expr(let_body(e), depth - 1));
        case expr_kind::MData:
            return lean::hash(r, hash_expr(mdata_expr(e), depth - 1));
        case expr_kind::Proj:
            return lean::hash(r, hash_expr(proj_expr(e), depth - 1));
        case expr_kind::BVar: case expr_kind::FVar: case expr_kind::MVar:
        case expr_kind::Sort: case expr_kind::Const: case expr_kind::Lit:
            return r;
        }
        lean_unreachable();
    }
public:
    /* Deliberately not noexcept, like expr_hash: libstdc++ then retains each
       node's hash instead of recomputing this fingerprint during rehashing. */
    size_t operator()(expr const & e) const { return static_cast<size_t>(hash_expr(e, 2)); }
};
}
