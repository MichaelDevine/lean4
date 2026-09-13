/* Run with the actual mpz implementation for each available integer backend. */
#include <cstdio>
#include <cstdlib>
#include <set>
#include "runtime/init_module.h"
#include "runtime/mpz.h"

using namespace lean;
static unsigned checks = 0;
static void require(bool condition) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "mpz content check %u failed\n", checks); std::exit(1); }
}
int main() {
    initialize_runtime_module();
    std::set<uint64> hashes;
    for (unsigned exponent : {0u, 1u, 31u, 32u, 63u, 64u, 65u, 127u, 128u, 255u, 256u, 1024u, 4096u}) {
        mpz p = mpz(2u).pow(exponent);
        for (mpz const & value : {p - 1u, p, p + 1u, -1 * p}) {
            mpz copy(value.to_string().c_str());
            require(copy == value);
            require(copy.content_hash() == value.content_hash());
            mpz moved(std::move(copy));
            require(moved.content_hash() == value.content_hash());
        }
        require(hashes.insert(p.content_hash()).second);
    }
    mpz high = mpz(2u).pow(1024), middle = mpz(2u).pow(512);
    require((high + 3u).content_hash() != (high + middle + 3u).content_hash());
    require((high + 3u).hash() == (high + middle + 3u).hash());
    require((high - high).content_hash() == mpz(0u).content_hash());
    require(((high + 3u) & mpz(7u)).content_hash() == mpz(3u).content_hash());
    std::printf("{\"checks\":%u,\"valid\":true}\n", checks);
    return 0;
}
