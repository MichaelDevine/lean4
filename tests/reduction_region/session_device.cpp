/* Additive actual-expression SYCL integration control. Requires a GPU. */
#include <iostream>
#include "runtime/init_module.h"
#include "kernel/reduction_session.h"
#include "library/reduction_sycl.h"
extern "C" void lean_initialize();

int main() {
    using namespace lean;
    using namespace lean::reduction;
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    reduction_sycl_backend gpu;
    expr type = mk_const(name("Nat"));
    expr value = mk_lit(lean::literal(19u));
    expr keep = mk_lambda(name("x"), type,
        mk_lambda(name("y"), type, mk_bvar(1), binder_info::Default), binder_info::Default);
    expr input = mk_app(mk_app(keep, value), mk_const(name("unused")));
    reduction_session session(input, 0, 0);
    if (session.advance(gpu) != outcome::need_arguments ||
        !is_bi_equal(session.reconstruct(), input)) return 1;
    session.resize_workspace(2, 0);
    if (session.advance(gpu) != outcome::need_bindings) return 1;
    session.resize_workspace(2, 2);
    if (session.advance(gpu) != outcome::complete || session.reconstruct() != value) return 1;
    std::cout << "Lean expression GPU session checks passed\n";
}
