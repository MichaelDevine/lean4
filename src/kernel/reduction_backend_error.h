/*
Copyright (c) 2026 Michael Devine. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
*/
#pragma once
#include <stdexcept>

namespace lean {
/** Device availability/execution failure, distinct from a mathematical result
    and from an unexpected implementation exception. Backend adapters translate
    their platform-specific errors to this portable host exception. */
class reduction_backend_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
}
