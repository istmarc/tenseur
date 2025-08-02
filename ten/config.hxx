#ifndef TENSEUR_CONFIG_HXX
#define TENSEUR_CONFIG_HXX

#include <optional>
#include <ostream>

namespace ten {

// Architecture
enum class arch { darwin, x64, unknown };

template <arch arch_type> struct is_supported_arch {
   static constexpr bool value = arch_type != arch::unknown;
};

constexpr arch get_arch_type() {
#if defined(__APPLE__)
   return Arch::darwin;
#endif
#if defined(__linux__)
   return arch::x64;
#endif
   return arch::unknown;
}
static constexpr arch arch_type = get_arch_type();

static_assert(is_supported_arch<arch_type>::value, "Unsuported architecture.");

// Verbose mode
/*MAYBE #ifndef TENSEUR_VERBOSE
#define TENSEUR_VERBOSE false
#endif*/
static bool is_verbose = false;

static void verbose(bool verbose_mode) {
   is_verbose = verbose_mode;
}

// Simd
enum class simd_backend { std, unknown };

#ifndef TENSEUR_SIMDBACKEND
#define TENSEUR_SIMDBACKEND simd_backend::std
#endif
static simd_backend simd_backend_type = TENSEUR_SIMDBACKEND;

// TODO Two simdvecLen default to 8 and 4 for floats, 4 and 2 for doubles
#ifndef TENSEUR_SIMDVECLEN_FLOAT
#define TENSEUR_SIMDVECLEN_FLOAT 32
#endif
static constexpr size_t simd_vecLen_float = TENSEUR_SIMDVECLEN_FLOAT;

#ifndef TENSEUR_SIMDVECLEN_DOUBLE
#define TENSEUR_SIMDVECLEN_DOUBLE 16
#endif
static constexpr size_t simd_vecLen_double = TENSEUR_SIMDVECLEN_DOUBLE;

} // namespace ten

#endif
