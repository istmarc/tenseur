#ifndef TA_KERNELS_STD_SIMD_BINARY_OPS_HXX
#define TA_KERNELS_STD_SIMD_BINARY_OPS_HXX

#include <experimental/bits/simd.h>
#include <experimental/simd>

#include <ten/config.hxx>
#include <ten/types.hxx>

namespace ten::kernels {

template <::ten::trig_operation kind, class A, class C>
static void trigOps(const A &a, C &c) {
   size_t n = a.size();
   constexpr size_t vlen = ::ten::simd_vecLen;
   using ::ten::binary_operation;
   using T = typename A::value_type;
   using vector_type = std::experimental::fixed_size_simd<float, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < n / vlen; i++) {
      // Load a and b
      size_t offset = i * vlen;
      vector_type a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type c_vec;
      switch (kind) {
      case trig_operation::sin:
         c_vec = ::std::sin(a_vec);
         break;
      case trig_operation::cos:
         c_vec = ::std::cos(a_vec);
         break;
      case trig_operation::tan:
         c_vec = ::std::tan(a_vec);
         break;
      }
      // Copy back
      c_vec.copy_to(c.data() + offset, alignment{});
   }
   for (size_t i = vlen * (n / vlen); i < n; i++) {
      switch (kind) {
      case trig_operation::sin:
         c[i] = ::std::sin(a[i]);
         break;
      case trig_operation::cos:
         c[i] = ::std::cos(a[i]);
         break;
      case trig_operation::tan:
         c[i] = ::std::tan(a[i]);
         break;
      }
   }
}
} // namespace ten::kernels

#endif
