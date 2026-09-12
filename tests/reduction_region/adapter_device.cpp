/* Actual-checker GPU control; unavailable-device fallback is reported separately. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "runtime/init_module.h"
#include "kernel/type_checker.h"
#include "library/closure_reduction_extension.h"
#include "library/reduction_sycl.h"
extern "C" void lean_initialize();
extern "C" lean_object * l_Lean_mkEmptyEnvironment(uint32_t);
extern "C" lean_object * lean_elab_environment_to_kernel_env(lean_object *);

int main(int argc, char ** argv) {
    using namespace lean;
    using namespace lean::reduction;
    bool expect_unavailable = argc == 2 && std::strcmp(argv[1], "--expect-unavailable") == 0;
    if (argc != 1 && !expect_unavailable) return 2;
    initialize_runtime_module(); lean_initialize(); lean_io_mark_end_initialization();
    auto created = l_Lean_mkEmptyEnvironment(0);
    if (!lean_io_result_is_ok(created)) return 1;
    auto raw = lean_io_result_get_value(created); lean_inc(raw); lean_dec(created);
    environment env(lean_elab_environment_to_kernel_env(raw));
    expr type = mk_sort(mk_level_one());
    expr value = mk_sort(mk_level_zero());
    expr identity = mk_lambda(name("x"), type, mk_bvar(0), binder_info::Default);
    expr input = mk_app(identity, value);
    auto admission = [](reduction_session const & s, outcome reason) {
        return optional<reduction_workspace>(reduction_workspace{
            s.argument_capacity() + (reason == outcome::need_arguments),
            s.binding_capacity() + (reason == outcome::need_bindings)});
    };
    reduction_sycl_backend gpu;
    closure_reduction_extension extension(gpu, admission);
    scope_reduction_extension enable(&extension);
    type_checker tc(env);
    if (tc.whnf(input) != value) return 1;
    if (expect_unavailable) {
        if (extension.last_attempt() != reduction_attempt::device_failure || extension.device_error().empty()) return 1;
        std::puts("unavailable-device CPU fallback verified; no GPU execution");
    } else {
        if (extension.last_attempt() != reduction_attempt::complete) return 1;
        std::puts("actual-checker GPU adapter checks passed");
    }
}
