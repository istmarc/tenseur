#ifndef TENSEUR_TESTS_TESTS
#define TENSEUR_TESTS_TESTS

#include <ten/types.hxx>
#include <gtest/gtest.h>
#include <type_traits>

namespace ten::tests {
// Compare the value type of two tensors
template <class A, class B>
testing::AssertionResult same_value_type(const A &a, const B &b) {
   using T = typename A::value_type;
   using R = typename B::value_type;
   if (!std::is_same_v<T, R>)
      return testing::AssertionFailure()
             << "Different value type " << ::ten::to_string<T>() << " and "
             << ::ten::to_string<R>() << std::endl;
   return testing::AssertionSuccess();
}

// Compare the shape of two tensors
template <class A, class B>
testing::AssertionResult same_shape(const A &a, const B &b) {
   if (A::rank() != B::rank())
      return testing::AssertionFailure() << "Different rank " << A::rank()
                                         << " and " << B::rank() << std::endl;

   if (a.size() != b.size())
      return testing::AssertionFailure() << "Different sizes " << a.size()
                                         << " and " << b.size() << std::endl;

   for (size_t i = 0; i < A::rank(); i++)
      if (a.dim(i) != b.dim(i))
         return testing::AssertionFailure()
                << "Different dimensions at index " << i << std::endl;
   return testing::AssertionSuccess();
}

// Compare the shape of two static tensors
template <class A, class B>
   requires(::ten::is_stensor<A>::value && ::ten::is_stensor<B>::value)
testing::AssertionResult same_shape(const A &a, const B &b) {
   if (A::rank() != B::rank())
      return testing::AssertionFailure() << "Different rank " << A::rank()
                                         << " and " << B::rank() << std::endl;

   if (A::static_size() != B::static_size())
      return testing::AssertionFailure()
             << "Different sizes " << A::static_size() << " and "
             << B::static_size() << std::endl;

   // TODO Compare all static dimensions
   return testing::AssertionSuccess();
}

// Compare the values of two floating point tensors
template <class A, class B>
   requires std::is_floating_point_v<typename A::value_type> &&
            std::is_floating_point_v<typename B::value_type>
testing::AssertionResult same_values(const A &a, const B &b,
                                     const double tol = 1e-3) {
   for (size_t i = 0; i < a.size(); i++) {
      auto val = std::abs(a[i] - b[i]);
      if (static_cast<double>(val) > tol)
         return testing::AssertionFailure()
                << "Different values at index " << i;
   }
   return testing::AssertionSuccess();
}

// Compare the values of two integral tensors
template <class A, class B>
   requires std::is_integral_v<typename A::value_type> &&
            std::is_integral_v<typename B::value_type>
testing::AssertionResult same_values(const A &a, const B &b) {
   for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i])
         return testing::AssertionFailure()
                << "Different values at index " << i;
   }
   return testing::AssertionSuccess();
}

// Compare two tensors
template <class A, class B>
   requires(::ten::is_dtensor<A>::value &&
            ::ten::is_dtensor<B>::value)
testing::AssertionResult equal(const A &a, const B &b, double eps = 1e-3) {
   testing::AssertionResult r_type = same_value_type(a, b);
   if (!r_type)
      return r_type;
   testing::AssertionResult r_shape = same_shape(a, b);
   if (!r_shape)
      return r_shape;
   testing::AssertionResult r_values = same_values(a, b, eps);
   if (!r_values)
      return r_values;

   return testing::AssertionSuccess();
}

// Compare two static tensors
template <class A, class B>
   requires(::ten::is_stensor<A>::value && ::ten::is_stensor<B>::value)
testing::AssertionResult equal(const A &a, const B &b, double eps = 1e-3) {
   testing::AssertionResult r_type = same_value_type(a, b);
   if (!r_type)
      return r_type;
   testing::AssertionResult r_shape = same_shape(a, b);
   if (!r_shape)
      return r_shape;
   testing::AssertionResult r_values = same_values(a, b, eps);
   if (!r_values)
      return r_values;

   return testing::AssertionSuccess();
}

} // namespace ten::tests

#endif
