#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <functional>
#include <initializer_list>
// #include <iostream>
#include <memory>
#include <type_traits>

#include <ten/kernels/host>
#include <ten/types.hxx>

namespace ten {
enum class binary_operation;
}

namespace ten::details {
template <class, class> struct common_type;

// Tensor, Tensor
template <Tensor A, Tensor B>
   requires(same_shape<A, B> && same_storage_order<A, B> &&
            same_storage<A, B> && same_allocator<A, B>)
struct common_type<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type, typename A::shape_type, A::storage_order(),
                     typename A::storage_type, typename A::allocator_type>;
};

// Scalar, Tensor
template <Scalar A, Tensor B>
//   requires(::ten::same_value_type<A, B>)
struct common_type<A, B> {
   using type = B;
};

// Tensor, Scalar
template <Tensor A, Scalar B>
//   requires(::ten::same_value_type<A, B>)
struct common_type<A, B> {
   using type = A;
};

template <class A, class B>
using common_type_t = typename common_type<A, B>::type;

template <class, class> struct mul_result;

template <class A, class B>
using mul_result_t = typename mul_result<A, B>::type;

// scalar * scalar
template<Scalar A, Scalar B>
struct mul_result<A, B> {
   using value_type = std::common_type_t<typename A::value_type, typename B::value_type>;
   using type = ::ten::scalar<value_type>;
};

// vector * vector
template <Vector A, Vector B>
   requires(same_shape<A, B> && same_storage_order<A, B> &&
            same_storage<A, B> && same_allocator<A, B>)
struct mul_result<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type, typename A::shape_type, A::storage_order(),
                     typename A::storage_type, typename A::allocator_type>;
};

// matrix * matrix
template <Matrix A, Matrix B>
   requires(same_storage_order<A, B> && same_storage<A, B> &&
            same_allocator<A, B>)
struct mul_result<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type,
                     ::ten::shape<A::shape_type::template static_dim<0>(),
                                  B::shape_type::template static_dim<1>()>,
                     A::storage_order(), typename A::storage_type,
                     typename A::allocator_type>;
};

// matrix * vector
template <Matrix A, Vector B>
   requires(same_storage_order<A, B> && same_storage<A, B> &&
            same_allocator<A, B>)
struct mul_result<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type, shape<A::shape_type::template static_dim<0>()>,
                     A::storage_order(), typename A::storage_type,
                     typename A::allocator_type>;
};

// scalar * tensor
template <Scalar A, Tensor B> struct mul_result<A, B> {
   using type = B;
};

/// scalar * unary_expr
template <Scalar A, UnaryExpr B> struct mul_result<A, B> {
   using type = std::remove_cvref_t<B>::output_type;
};

/// scalar * binary_expr
template <Scalar A, BinaryExpr B> struct mul_result<A, B> {
   using type = std::remove_cvref_t<B>::output_type;
};

/// UnaryExpr * vector
template <UnaryExpr A, Vector B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * vector
template <BinaryExpr A, Vector B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// UnaryExpr * matrix
template <UnaryExpr A, Matrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// matrix * UnaryExpr
template <Matrix A, UnaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// BinaryExpr * Matrix
template <BinaryExpr A, Matrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// Matrix * BinaryExpr
template <Matrix A, BinaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// UnaryExpr * tensor
template <UnaryExpr A, Tensor B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * tensor
template <BinaryExpr A, Tensor B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

// dynamic reshape result
template <class A, class Shape> struct reshape_result {
   using type =
       ranked_tensor<typename A::value_type, Shape, A::storage_order(),
                     typename A::storage_type, typename A::allocator_type>;
};

// static transpose result
template <class A, class Shape> struct static_transpose_result {
   static_assert(Shape::is_static(), "Shape must be static.");
   static_assert(Shape::rank() == 2, "Shape rank must be 2.");

   using type = ranked_tensor<typename A::value_type,
                              ::ten::shape<Shape::template static_dim<1>(),
                                           Shape::template static_dim<0>()>,
                              A::storage_order(), typename A::storage_type,
                              typename A::allocator_type>;
};

// dynamic transpose result
template <class A, class Shape> struct transpose_result {
   using type =
       ranked_tensor<typename A::value_type, Shape, A::storage_order(),
                     typename A::storage_type, typename A::allocator_type>;
};

} // namespace ten::details

namespace ten::functional {
////////////////////////////////////////////////////////////////////////////////
// Functions types
template <bool params = false, bool with_shape = false> struct func {};

template <class Func> struct has_params {
   static constexpr bool value = std::is_base_of_v<func<true, true>, Func> ||
                                 std::is_base_of_v<func<true, false>, Func>;
};

template <class Func> struct has_shape {
   static constexpr bool value = std::is_base_of_v<func<true, true>, Func> ||
                                 std::is_base_of_v<func<false, true>, Func>;
};

////////////////////////////////////////////////////////////////////////////////
// Unary functions

/// Square root
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
   || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqrt : func<> {
   static constexpr std::string name() { return std::string("sqrt"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      if constexpr (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>) {
         y.value() = std::sqrt(x.value());
      }
      if constexpr (::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::sqrt(static_cast<output_value_type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      y = 1 / (2* std::sqrt(x));
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = 1 / (2 * std::sqrt(x[i]));
      }
   }
};

/// Square
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value ) && ::ten::is_tensor<Y>::value)
      || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqr : func<> {
   static constexpr std::string name() { return std::string("sqr"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      if constexpr (::ten::is_scalar<X>::value) {
         y.value() = x.value() * x.value();
      }
   if constexpr (::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = static_cast<output_value_type>(x[i]) * static_cast<output_value_type>(x[i]);
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      y = output_value_type(2) * x;
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i< x.size() ; i++) {
         y[i] = output_value_type(2) * x[i];
      }
   }
};

/// absolute value
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
   || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct abs : func<> {
   static constexpr std::string name() { return std::string("abs"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {

      if constexpr (::ten::is_scalar_v<X>) {
         y.value() = std::abs(x.value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::abs(static_cast<value_type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      if (x>= 0) {
         y = 1;
      } else {
         y = -1;
      }
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         if (x[i]>=0) {
            y[i] = 1;
         } else {
            y[i] = -1;
         }
      }
   }
};

/// Power
template <class X, class Y>
   requires( ((ten::is_tensor<X>::value || ten::is_column<X>::value ||
             ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
      (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct pow : func<true> {
   static constexpr std::string name() { return std::string("pow"); }

 private:
   double _n;

 public:
   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   explicit pow(double n) : _n(n) {}

   void operator()(const X &x, Y &y) const {
      using value_type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>) {
         y.value() = std::pow(x.value(), _n);
      }
      if constexpr (::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::pow(static_cast<value_type>(x[i]), _n);
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      y = static_cast<output_value_type>(_n) * std::pow(x, _n-1);
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = static_cast<output_value_type>(_n) * std::pow(x[i], _n-1);
      }
   }
};

/// Minimum
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_scalar<Y>::value)
struct min : func<> {
   static constexpr std::string name() { return std::string("min"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename X::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::min(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename X::shape_type
   output_shape(const typename X::shape_type &left) {
      return left;
   }
};

/// Maximum
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_scalar<Y>::value)
struct max : func<> {
   static constexpr std::string name() { return std::string("max"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::max(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename X::shape_type
   output_shape(const typename X::shape_type &left) {
      return left;
   }
};

/// Sum
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_scalar<Y>::value)
struct sum : func<> {
   static constexpr std::string name() { return std::string("sum"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      type res = 0.;
      for (size_t i = 0; i < a.size(); i++) {
         res += static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Cumulative sum
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct cum_sum : func<> {
   static constexpr std::string name() { return std::string("cum_sum"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      b[0] = static_cast<type>(a[0]);
      for (size_t i = 1; i < a.size(); i++) {
         b[i] = static_cast<type>(a[i]) + b[i - 1];
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Prod
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_scalar<Y>::value)
struct prod : func<> {
   static constexpr std::string name() { return std::string("prod"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      type res = 1.;
      for (size_t i = 0; i < a.size(); i++) {
         res *= static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename X::shape_type
   output_shape(const typename X::shape_type &left) {
      return left;
   }
};

// TODO use simd for cos, sin and tan

/// Sine
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sin : func<> {
   static constexpr std::string name() { return std::string("sin"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sin(static_cast<type>(a[i]));
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   template<typename T>
   T gradient(T x) {
      return std::cos(x);
   }
};

/// asin
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct asin : func<> {
   static constexpr std::string name() { return std::string("asin"); }

   using output_type = Y;

   void operator()(const X &x, Y &y) {
      using type = typename Y::value_type;
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = std::asin(static_cast<type>(x[i]));
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Sinh
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct sinh : func<> {
   static constexpr std::string name() { return std::string("asin"); }

   using output_type = Y;

   void operator()(const X &x, Y &y) {
      using type = typename Y::value_type;
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = std::sinh(static_cast<type>(x[i]));
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Cosine
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct cos : func<> {
   static constexpr std::string name() { return std::string("cos"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      if constexpr (::ten::is_scalar_v<X>) {
         y.value() = std::cos(x.value());
      }
      if constexpr(::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::cos(static_cast<output_value_type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      y = -std::sin(x);
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = -std::sin(x[i]);
      }
   }
};

/// acos
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct acos : func<> {
   static constexpr std::string name() { return std::string("acos"); }

   using output_type = Y;

   void operator()(const X &x, Y &y) {
      using type = typename Y::value_type;
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = std::acos(static_cast<type>(x[i]));
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// cosh
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct cosh : func<> {
   static constexpr std::string name() { return std::string("cosh"); }

   using output_type = Y;

   void operator()(const X &a, Y &b) {
      using type = typename Y::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::cosh(static_cast<type>(a[i]));
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Tangent
template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tan : func<> {
   static constexpr std::string name() { return std::string("tan"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      if constexpr (::ten::is_scalar_v<X>) {
         y.value() = std::tan(x.value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::tan(static_cast<output_value_type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      value_type z = std::tan(x);
      y = 1 + z*z;
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         value_type z = std::tan(x[i]);
         y[i] = 1 + z*z;
      }
   }
};

template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct atan : func<> {
   static constexpr std::string name() { return std::string("atan"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         y.value() = std::atan(static_cast<type>(x.value()));
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::atan(static_cast<type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, output_value_type& y) {
      y = 1 / (1 + x*x);
   }

   void gradient(const X& x, Y& y) {
      for (size_t i = 0; i < x.size(); i++) {
         y[i] = 1 / (1 + x[i]*x[i]);
      }
   }
};

template <class X, class Y>
   requires( ((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) || (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tanh : func<> {
   static constexpr std::string name() { return std::string("tanh"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(const X &x, Y &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         y.value() = std::tanh(x.value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::tanh(static_cast<type>(x[i]));
         }
      }
   }

   static typename Y::shape_type
   output_shape(const typename Y::shape_type &right) {
      return right;
   }
};

/// Exponential
template <class A, class B>
   requires( ((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value) || (::ten::is_scalar_v<A> && ::ten::is_scalar_v<B>))
struct exp : func<> {
   static constexpr std::string name() { return std::string("exp"); }

   using value_type = A::value_type;
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      if constexpr (::ten::is_scalar_v<A>) {
         b.value() = std::exp(a.value());
      }
      if constexpr (::ten::is_tensor_v<B>) {
         for (size_t i = 0; i < a.size(); i++) {
            b[i] = std::exp(static_cast<type>(a[i]));
         }
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, value_type& y) {
      y = std::exp(x);
   }

   void gradient(const A& a, B& b) {
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::exp(a[i]);
      }
   }
};

/// Natural logarithm
template <class A, class B>
   requires( ((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value) || (::ten::is_scalar_v<A> && ::ten::is_scalar_v<B>))
struct log : func<> {
   static constexpr std::string name() { return std::string("log"); }

   using value_type = A::value_type;
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      if constexpr (::ten::is_scalar_v<A>) {
         b.value() = std::log(a.value());
      }
      if constexpr (::ten::is_tensor_v<B>) {
         for (size_t i = 0; i < a.size(); i++) {
            b[i] = std::log(static_cast<type>(a[i]));
         }
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, value_type& y) {
      y = value_type(1) / x;
   }

   void gradient(const A& a, B& b) {
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = value_type(1) / a[i];
      }
   }
};

/// Logarithm
template <class A, class B>
   requires( ((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value) || (::ten::is_scalar_v<A> && ::ten::is_scalar_v<B>))
struct log10 : func<> {
   static constexpr std::string name() { return std::string("log10"); }
   using value_type = A::value_type;
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      if constexpr (::ten::is_scalar_v<A>) {
            b.value() = std::log10(a.value());
      }
      if constexpr (::ten::is_tensor_v<B>) {
         for (size_t i = 0; i < a.size(); i++) {
            b[i] = std::log10(static_cast<type>(a[i]));
         }
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }

   void gradient(const value_type& x, value_type& y) {
      y = value_type(1) / (x * std::log(value_type(10)));
   }

   void gradient(const A& a, B& b) {
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = value_type(1) / (a[i] * std::log(value_type(10)));
      }
   }
};

/// Floor
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct floor : func<> {
   static constexpr std::string name() { return std::string("floor"); }

   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::floor(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Ceil
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct ceil : func<> {
   static constexpr std::string name() { return std::string("ceil"); }

   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::ceil(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

////////////////////////////////////////////////////////////////////////////////
// Binary functions (Add, Sub, Mul and Div)

/// Binary function
template <::ten::binary_operation Kind> struct binary_func {

   template <class A, class B, class C>
   struct func : ::ten::functional::func<> {
      static constexpr std::string name() {
         if constexpr (Kind == ::ten::binary_operation::add) {
            return std::string("add");
         }
         if constexpr (Kind == ::ten::binary_operation::sub) {
            return std::string("sub");
         }
         if constexpr (Kind == ::ten::binary_operation::mul) {
            return std::string("mul");
         }
         if constexpr (Kind == ::ten::binary_operation::div) {
            return std::string("div");
         }
      }

      using output_type = C;

      static constexpr typename C::shape_type
      output_shape(const typename A::shape_type &/*left*/,
                   const typename B::shape_type &right) {
         // FIXME Maybe check that left == right
         typename C::shape_type s(right);
         return s;
      }

      // static auto output_shape(const A &a, const B &b) { return a.shape(); }

      void operator()(const A &left, const B &right, C &result) {
         ::ten::kernels::binary_ops<Kind>(left, right, result);
      }

      void gradient_left(const A& /*x*/, const B& y, C& z) {
         if constexpr ((Kind == ::ten::binary_operation::add)
         || (Kind == ::ten::binary_operation::sub)) {
            for (size_t i = 0; i < z.size();i++) {
               z[i] = 1;
            }
         }
         if (Kind == ::ten::binary_operation::mul) {
            for (size_t i = 0; i< z.size();i++) {
               z[i] = y[i];
            }
         }
         if (Kind == ::ten::binary_operation::div) {
            for (size_t i = 0; i< z.size();i++) {
               z[i] = 1 / y[i];
            }
         }
      }

      void gradient_right(const A& x, const B& y, C& z) {
         if constexpr (Kind == ::ten::binary_operation::add) {
            for (size_t i = 0; i < z.size();i++) {
               z[i] = 1;
            }
         }
         if constexpr (Kind == ::ten::binary_operation::sub) {
            for (size_t i = 0; i < z.size();i++) {
               z[i] = -1;
            }
         }
         if (Kind == ::ten::binary_operation::mul) {
            for (size_t i = 0; i< z.size();i++) {
               z[i] = x[i];
            }
         }
         if (Kind == ::ten::binary_operation::div) {
            for (size_t i = 0; i< z.size();i++) {
               z[i] = - x[i] / (y[i]*y[i]);
            }
         }
      }
   };
};

/// Binary function with scalar
template <::ten::binary_operation Kind> struct scalar_left_binary_func {

   template <Scalar A, class B, class C>
   struct func : ::ten::functional::func<> {
      static constexpr std::string name() {
         if constexpr (Kind == ::ten::binary_operation::add) {
            return std::string("add");
         }
         if constexpr (Kind == ::ten::binary_operation::sub) {
            return std::string("sub");
         }
         if constexpr (Kind == ::ten::binary_operation::mul) {
            return std::string("mul");
         }
         if constexpr (Kind == ::ten::binary_operation::div) {
            return std::string("div");
         }
      }

      using output_type = C;

      static constexpr typename C::shape_type
      output_shape( // const typename A::shape_type &left,
          const typename B::shape_type &right) {
         // FIXME Maybe check that left == right
         typename C::shape_type s(right);
         return s;
      }

      static auto output_shape(/*const A &a, */ const B &b) {
         return b.shape();
      }

      void operator()(const A &left, const B &right, C &result) {
         /// FIXME use broadcast
         if constexpr (B::is_static()) {
            B x;
            for (size_t i = 0; i < x.size(); i++) {
               x[i] = left.value();
            }
            ::ten::kernels::binary_ops<Kind>(x, right, result);
         } else {
            B x(right.shape());
            for (size_t i = 0; i < x.size(); i++) {
               x[i] = left.value();
            }
            ::ten::kernels::binary_ops<Kind>(x, right, result);
         }
      }
   };
};

/// Binary function with scalar right type
template <::ten::binary_operation Kind> struct scalar_right_binary_func {

   template <class A, Scalar B, class C>
   struct func : ::ten::functional::func<> {
      static constexpr std::string name() {
         if constexpr (Kind == ::ten::binary_operation::add) {
            return std::string("add");
         }
         if constexpr (Kind == ::ten::binary_operation::sub) {
            return std::string("sub");
         }
         if constexpr (Kind == ::ten::binary_operation::mul) {
            return std::string("mul");
         }
         if constexpr (Kind == ::ten::binary_operation::div) {
            return std::string("div");
         }
      }

      using output_type = C;

      static constexpr typename C::shape_type
      output_shape(const typename A::shape_type &left) {
         typename C::shape_type s(left);
         return s;
      }

      static auto output_shape(const A &a) { return a.shape(); }

      void operator()(const A &left, const B &right, C &result) {
         /// FIXME use broadcast
         if constexpr (A::is_static()) {
            A x;
            for (size_t i = 0; i < x.size(); i++) {
               x[i] = right.value();
            }
            ::ten::kernels::binary_ops<Kind>(left, x, result);
         } else {
            A x(left.shape());
            for (size_t i = 0; i < x.size(); i++) {
               x[i] = right.value();
            }
            ::ten::kernels::binary_ops<Kind>(left, x, result);
         }
      }
   };
};

template <class A, class B, class C> struct mul : func<> {};

// vector * vector
template <Vector A, Vector B, Vector C>
struct mul<A, B, C> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = C;

   static constexpr typename C::shape_type
   output_shape(const typename A::shape_type &left,
                const typename B::shape_type &right) {
      // FIXME Maybe check that left == right
      typename C::shape_type s(right);
      return s;
   }

   void operator()(const A &left, const B &right, C &result) {
      ::ten::kernels::binary_ops<::ten::binary_operation::mul>(left, right,
                                                               result);
   }
};

// matrix * matrix
template <Matrix A, Matrix B, Matrix C> struct mul<A, B, C> : func<> {

   static constexpr std::string name() { return std::string("mul"); }

   using output_type = C;

   static typename C::shape_type
   output_shape(const typename A::shape_type &left,
                const typename B::shape_type &right) {
      std::initializer_list<size_type> &&dims = {left.dim(0), right.dim(1)};
      typename C::shape_type s(std::move(dims));
      return s;
   }

   void operator()(const A &left, const B &right, C &result) {
      kernels::mul(left, right, result);
   }
};

// matrix * vector
template <Matrix A, Vector B, Vector C>
struct mul<A, B, C> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = C;

   static typename C::shape_type
   output_shape(const typename A::shape_type &left,
                const typename B::shape_type &right) {
      std::initializer_list<size_type> &&dims = {left.dim(0)};
      typename C::shape_type s(std::move(dims));
      return s;
   }

   void operator()(const A &left, const B &right, C &result) {
      kernels::mul(left, right, result);
   }
   //};
};

// scalar * tensor
template <Scalar A, Tensor B, Tensor C> struct mul<A, B, C> : func<> {

   static constexpr std::string name() { return std::string("mul"); }

   using output_type = C;

   static typename C::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }

   void operator()(const A &left, const B &right, C &result) {
      // TODO Broadcasting
      size_t n = result.size();
      using value_type = typename C::value_type;
      for (size_t i = 0; i < n; i++) {
         result[i] = static_cast<value_type>(left.value()) *
                     static_cast<value_type>(right[i]);
      }
   }
};

/*
// UnaryExpr * matrix
template <UnaryExpr X, Matrix Y, Matrix Z>
// FIXME REquire X::output_type to be a matrix
struct mul<X, Y, Z>  {
   // using evaluated_type = std::remove_cvref_t<X>::output_type;
   //  matrix * matrix
   template <Matrix A, Matrix B, Matrix C>
   using func = mul<A, B, C>::template func<A, B, C>;
};

// BinaryExpr * matrix
template <BinaryExpr X, Matrix Y, Matrix Z>
// FIXME REquire X::output_type to be a matrix
struct mul<X, Y, Z> {
   // using evaluated_type = std::remove_cvref_t<X>::output_type;
   //  matrix * matrix
   template <Matrix A, Matrix B, Matrix C>
   using func = mul<A, B, C>::template func<A, B, C>;
};*/

// UnaryExpr * tensor
/*template <UnaryExpr X, Tensor Y, Tensor  Z>
struct mul<X, Y, Z> {
   using evaluated_type = std::remove_cvref_t<X>::evaluated_type::node_type;
   template <Scalar A, Tensor B, Tensor C>
   using func = mul<A, B, C>::template func<A, B, C>;
};*/

////////////////////////////////////////////////////////////////////////////////
// Reshape

// Reshape to static
template <class Shape> struct static_reshape {
   static constexpr std::string name() { return std::string("static_reshape"); }

   static_assert(Shape::is_static(), "Shape must be static");

   template <class A, class B>
      requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
                ::ten::is_row<A>::value) &&
               ::ten::is_tensor<B>::value)
   struct func : ::ten::functional::func<> {
      static_assert(A::is_static(),
                    "A must be static tensor or static column or static row.");

      using output_type = B;

      void operator()(const A &left, B &right) {
         // Does not copy the data
         right = B(left.node(), left.format());
      }
   };
};

template <size_type... __dims>
using dims_static_reshape = static_reshape<::ten::shape<__dims...>>;

// Dynamic reshape
// Shape is the target shape type
template <class Shape> struct dynamic_reshape {

   template <class A, class B>
      requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
                ::ten::is_row<A>::value) &&
               ::ten::is_tensor<B>::value)
   struct func : ::ten::functional::func<true, true> {
      static constexpr std::string name() { return std::string("reshape"); }

    public:
      using output_type = B;

    private:
      Shape _shape;

    public:
      func(const Shape &s) : _shape(s) {}
      func(Shape &&s) : _shape(std::move(s)) {}

      typename B::shape_type output_shape(const typename A::shape_type &) {
         return _shape;
      }

      void operator()(const A &left, B &right) {
         // Does not copy the data
         right = B(left.node(), _shape, left.format());
      }
   };
};

////////////////////////////////////////////////////////////////////////////////
// Transpose a matrix

// Reshape to static
template <class Shape> struct static_transpose {
   static_assert(Shape::is_static(), "Shape must be static.");
   static_assert(Shape::rank() == 2, "Shape rank must be 2.");

   template <class A, class B> struct func : ::ten::functional::func<> {
      static constexpr std::string name() {
         return std::string("static_transpose");
      }

      using output_type = B;

      void operator()(const A &left, B &right) {
         using shape_type = B::shape_type;
         // FIXME use auto storage =
         // std::make_shared<storage_type>(storage_type());
         // right = B(storage);
         // right = B();
         // copy transposed data
         size_type m = A::shape_type::template static_dim<0>();
         size_type n = A::shape_type::template static_dim<1>();
         for (size_type i = 0; i < n; i++) {
            for (size_type j = 0; j < m; j++) {
               right(i, j) = left(j, i);
            }
         }
      }
   };
};

/*
template <size_type rows, size_type cols>
using DimsStaticTranspose = StaticTranspose<::ten::Shape<cols, rows>>;*/

// Dynamic transpose
template <class Shape> struct dynamic_transpose {

   template <class A, class B> struct func : ::ten::functional::func<true> {
      static_assert(Shape::is_dynamic(), "Shape must be dynamic");
      // FIXME Currently defined only for matrices
      static_assert(Shape::rank() == 2, "Shape rank must be 2.");

      static constexpr std::string name() { return std::string("transpose"); }

      using output_type = B;

      typename B::shape_type output_shape(const typename A::shape_type &s) {
         typename B::shape_type res({s.dim(1), s.dim(0)});
         return res;
      }

      void operator()(const A &left, B &right) {
         using shape_type = B::shape_type;

         size_type m = left.dim(0);
         size_type n = left.dim(1);
         // shape_type s({left.dim(1), left.dim(0)});
         //
         for (size_type i = 0; i < n; i++) {
            for (size_type j = 0; j < m; j++) {
               right(i, j) = left(j, i);
            }
         }
      }
   };
};

} // namespace ten::functional

#endif
