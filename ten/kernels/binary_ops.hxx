#ifndef TENSEUR_KERNELS_STD_SIMD_BINARY_OPS_HXX
#define TENSEUR_KERNELS_STD_SIMD_BINARY_OPS_HXX

#include <experimental/bits/simd.h>
#include <experimental/simd>

#include <ten/config.hxx>
#include <ten/types.hxx>

namespace ten::kernels {

template<::ten::binary_operation kind, typename T, class A, class B, class C>
static void binary_ops_32(const A &a, const B &b, C &c) {
   size_t n = a.size();
   constexpr size_t vlen = ::ten::simd_vecLen_float;
   using ::ten::binary_operation;
   using vector_type = std::experimental::fixed_size_simd<T, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   // max vlen size = 32
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

   // 16
   constexpr size_t vlen16 = 16;
   using vector_type16 = std::experimental::fixed_size_simd<T, vlen16>;
   size_t j = vlen * (n / vlen);
   size_t rest = n - j;
   if (rest >= vlen16) {
      size_t offset = j;
      vector_type16 a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type16 b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type16 c_vec;
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
      // copy back data
      c_vec.copy_to(c.data() + offset, alignment{});
      // Iter j
      j += vlen16;
   }

   // 8
   rest = n - j;
   constexpr size_t vlen8 = 8;
   using vector_type8 = std::experimental::fixed_size_simd<T, vlen8>;
   if (rest >= vlen8) {
      size_t offset = j;
      vector_type8 a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type8 b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type8 c_vec;
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
      // copy back data
      c_vec.copy_to(c.data() + offset, alignment{});
      // Iter j
      j += vlen8;
   }

   // 4
   rest = n - j;
   constexpr size_t vlen4 = 4;
   using vector_type4 = std::experimental::fixed_size_simd<T, vlen4>;
   if (rest >= vlen4) {
      size_t offset = j;
      vector_type4 a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type4 b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type4 c_vec;
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
      // copy back data
      c_vec.copy_to(c.data() + offset, alignment{});
      // Iter j
      j += vlen4;
   }

   // Rest
   for (size_t i = j; i < n; i++) {
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

template <::ten::binary_operation kind, typename T, class A, class B, class C>
static void binary_ops_16(const A &a, const B &b, C &c) {
   size_t n = a.size();
   constexpr size_t vlen = ::ten::simd_vecLen_double;
   using ::ten::binary_operation;
   using vector_type = std::experimental::fixed_size_simd<T, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   // max vlen size = 16
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

   // 8
   size_t j = vlen * (n/vlen);
   size_t rest = n - j;
   constexpr size_t vlen8 = 8;
   using vector_type8 = std::experimental::fixed_size_simd<T, vlen8>;
   if (rest >= vlen8) {
      size_t offset = j;
      vector_type8 a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type8 b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type8 c_vec;
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
      // copy back data
      c_vec.copy_to(c.data() + offset, alignment{});
      // Iter j
      j += vlen8;
   }

   // 4
   rest = n - j;
   constexpr size_t vlen4 = 4;
   using vector_type4 = std::experimental::fixed_size_simd<T, vlen4>;
   if (rest >= vlen4) {
      size_t offset = j;
      vector_type4 a_vec;
      a_vec.copy_from(a.data() + offset, alignment{});
      vector_type4 b_vec;
      b_vec.copy_from(b.data() + offset, alignment{});
      // c_vec = a_vec ops b_vec
      vector_type4 c_vec;
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
      // copy back data
      c_vec.copy_to(c.data() + offset, alignment{});
      // Iter j
      j += vlen4;
   }

   // Rest
   for (size_t i = j; i < n; i++) {
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
   binary_ops_32<kind, float>(a, b, c);
}

/// Binary operation for double
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_double<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_16<kind, double>(a, b, c);
}

/// Binary operation for int32
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_int32<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_32<kind, int32_t>(a, b, c);
}

/// Binary operation for uint32
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_uint32<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_32<kind, uint32_t>(a, b, c);
}

/// Binary operation for int64
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_int64<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_16<kind, int64_t>(a, b, c);
}

/// Binary operation for uint64
template <::ten::binary_operation kind, class A, class B, class C>
requires(::ten::is_uint64<typename C::value_type>::value)
static void binary_ops(const A &a, const B &b, C &c) {
   binary_ops_16<kind, uint64_t>(a, b, c);
}

// TODO complex<float> and complex<double>

} // namespace ten::kernels

#endif
