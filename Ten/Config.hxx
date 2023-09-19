#ifndef TENSEUR_CONFIG_HXX
#define TENSEUR_CONFIG_HXX

#include <optional>
#include <ostream>

namespace ten {

// Architecture
enum class Arch { darwin, x64, unknown };

template <Arch arch> struct is_supported_arch {
   static constexpr bool value = arch != Arch::unknown;
};

constexpr Arch get_arch_type() {
#if defined(__APPLE__)
   return Arch::darwin;
#endif
#if defined(__linux__)
   return Arch::x64;
#endif
   return Arch::unknown;
}
static constexpr Arch arch = get_arch_type();

static_assert(is_supported_arch<arch>::value, "Unsuported architecture.");

// Simd
enum class SimdBackend { std, unknown };

#ifndef TENSEUR_SIMDBACKEND
#define TENSEUR_SIMDBACKEND SimdBackend::std
#endif
static SimdBackend simdBackend = TENSEUR_SIMDBACKEND;

// TODO Two simdvecLen default to 8 and 4 for floats, 4 and 2 for doubles
#ifndef TENSEUR_SIMDVECLEN
#define TENSEUR_SIMDVECLEN 4
#endif
static constexpr size_t simdVecLen = TENSEUR_SIMDVECLEN;

} // namespace ten

#endif
