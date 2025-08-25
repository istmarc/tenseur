#ifndef TENSEUR_KERNELS_STD_SIMD_BINARY_OPS_HXX
#define TENSEUR_KERNELS_STD_SIMD_BINARY_OPS_HXX

#include <experimental/bits/simd.h>
#include <experimental/simd>

#include <ten/config.hxx>
#include <ten/types.hxx>

namespace ten::kernels {

template <::ten::binary_operation kind, typename T, class A, class B, class C>
static void binary_ops_simd(const A &a, const B &b, C &c) {
   static_assert(std::is_same_v<typename C::value_type, T>);
   static_assert(std::is_same_v<typename A::value_type, T>);
   static_assert(std::is_same_v<typename B::value_type, T>);

   size_t n = a.size();
   using ::ten::binary_operation;
   using vector_type = std::experimental::simd<T>;
   constexpr size_t vlen = vector_type::size();
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < n / vlen; i++) {
      // Load a and b
      size_t offset = i * vlen;
      vector_type a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type c_vec;
      switch (kind) {
      case binary_operation::add:
         c_vec = a_vec + b_vec;
         break;
      case binary_operation::sub:
         c_vec = a_vec - b_vec;
         break;
      case binary_operation::div:
         c_vec = a_vec / b_vec;
         break;
      case binary_operation::mul:
         c_vec = a_vec * b_vec;
         break;
      }
      // Copy back
      c_vec.copy_to(c.data() + offset, alignment{});
   }
   // Rest
   for (size_t i = vlen*(n/vlen); i < n; i++) {
      switch(kind) {
      case binary_operation::add:
         c[i] = a[i] + b[i];
         break;
      case binary_operation::sub:
         c[i] = a[i] - b[i];
         break;
      case binary_operation::div:
         c[i] = a[i] / b[i];
         break;
      case binary_operation::mul:
         c[i] = a[i] * b[i];
         break;
      }
   }
}

/// Binary operation for float
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_float<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, float>(a, b, c);
}

/// Binary operation for double
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_double<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, double>(a, b, c);
}

/// Binary operation for int32
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_int32<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, int32_t>(a, b, c);
}

/// Binary operation for uint32
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_uint32<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, uint32_t>(a, b, c);
}

/// Binary operation for int64
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_int64<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, int64_t>(a, b, c);
}

/// Binary operation for uint64
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_uint64<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_simd<kind, uint64_t>(a, b, c);
}

// TODO complex<float> and complex<double>

} // namespace ten::kernels

#endif
