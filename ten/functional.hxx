#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <functional>
#include <initializer_list>
//#include <iostream>
#include <memory>
#include <type_traits>

#include <ten/types.hxx>
#include <ten/kernels/host>

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
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct sqrt : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sqrt(static_cast<value_type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Square
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct sqr : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = static_cast<value_type>(a[i]) * static_cast<value_type>(a[i]);
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// absolute value
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct abs : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::abs(static_cast<value_type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Power
template <class A, class B>
   requires((ten::is_tensor<A>::value || ten::is_column<A>::value ||
             ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct pow : func<true> {
 private:
   double _n;

 public:
   using output_type = B;

   explicit pow(double n) : _n(n) {}

   void operator()(const A &a, B &b) const {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::pow(static_cast<value_type>(a[i]), _n);
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Minimum
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_scalar<B>::value)
struct min : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename A::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::min(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename A::shape_type
   output_shape(const typename A::shape_type &left) {
      return left;
   }
};

/// Maximum
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_scalar<B>::value)
struct max : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::max(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename A::shape_type
   output_shape(const typename A::shape_type &left) {
      return left;
   }
};
/// Sum
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_scalar<B>::value)
struct sum : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      type res = 0.;
      for (size_t i = 0; i < a.size(); i++) {
         res += static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Cumulative sum
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct cum_sum : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      b[0] = static_cast<type>(a[0]);
      for (size_t i = 1; i < a.size(); i++) {
         b[i] = static_cast<type>(a[i]) + b[i - 1];
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Prod
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_scalar<B>::value)
struct prod : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      type res = 1.;
      for (size_t i = 0; i < a.size(); i++) {
         res *= static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename A::shape_type
   output_shape(const typename A::shape_type &left) {
      return left;
   }
};

// TODO use simd for cos, sin and tan

/// Sine
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct sin : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sin(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// asin
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct asin : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::asin(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Sinh
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct sinh : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sinh(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Cosine
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct cos : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::cos(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// acos
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct acos : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::acos(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// cosh
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct cosh : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::cosh(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Tangent
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct tan : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::tan(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct atan : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::atan(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct tanh : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::tanh(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Exponential
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct exp : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::exp(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Natural logarithm
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct log : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::log(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Logarithm
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct log10 : func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::log10(static_cast<type>(a[i]));
      }
   }

   static typename B::shape_type
   output_shape(const typename B::shape_type &right) {
      return right;
   }
};

/// Floor
template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value ||
             ::ten::is_row<A>::value) &&
            ::ten::is_tensor<B>::value)
struct floor : func<> {
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

      using output_type = C;

      static constexpr typename C::shape_type
      output_shape(const typename A::shape_type &left,
                   const typename B::shape_type &right) {
         // FIXME Maybe check that left == right
         typename C::shape_type s(left);
         return s;
      }

      static auto output_shape(const A &a, const B &b) { return a.shape(); }

      void operator()(const A &left, const B &right, C &result) {
         ::ten::kernels::binary_ops<Kind>(left, right, result);
      }
   };
};

/// Binary function with scalar
template <::ten::binary_operation Kind> struct scalar_left_binary_func {

   template <Scalar A, class B, class C>
   struct func : ::ten::functional::func<> {

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

template <class A, class B, class C>
// = typename details::mul_result<A, B>::type>
struct mul;

// vector * vector
template <Vector X, Vector Y, Vector Z> struct mul<X, Y, Z> {

   template <Vector A, Vector B, Vector C>
   using func = binary_func<::ten::binary_operation::mul>::func<A, B, C>;
};

// matrix * matrix
template <Matrix X, Matrix Y, Matrix Z> struct mul<X, Y, Z> {

   template <Matrix A, Matrix B, Matrix C>
   struct func : ::ten::functional::func<> {
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
};

// matrix * vector
template <Matrix X, Vector Y, Vector Z> struct mul<X, Y, Z> {

   template <Matrix A, Vector B, Vector C>
   struct func : ::ten::functional::func<> {
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
   };
};

// scalar * tensor
template <Scalar X, Tensor Y, Tensor Z> struct mul<X, Y, Z> {

   template <Scalar A, Tensor B, Tensor C>
   struct func : ::ten::functional::func<> {
      using output_type = C;

      static typename C::shape_type
      output_shape(const typename B::shape_type &right) {
         return right;
      }

      void operator()(const A &left, const B &right, C &result) {
         size_t n = result.size();
         using value_type = typename C::value_type;
         for (size_t i = 0; i < n; i++) {
            result[i] = static_cast<value_type>(left.value()) *
                        static_cast<value_type>(right[i]);
         }
      }
   };
};

// UnaryExpr * matrix
template <UnaryExpr X, Matrix Y, Matrix Z>
// FIXME REquire X::output_type to be a matrix
struct mul<X, Y, Z> {
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
};

// UnaryExpr * tensor
/*template <UnaryExpr X, Tensor Y, Tensor  Z>
struct mul<X, Y, Z> {
   using evaluated_type = std::remove_cvref_t<X>::evaluated_type::node_type;
   template <Scalar A, Tensor B, Tensor C>
   using func = mul<A, B, C>::template func<A, B, C>;
};*/

////////////////////////////////////////////////////////////////////////////////
/// Other BLAS functions

// C <- alpha * X * Y + beta * C
/*template<class A, class B, class C, class D = typename
::ten::functional::details::mul_result<A, B>::type> class gemm :
::ten::functional::func<true> { private: T _alpha; T _beta; public: gemm(const T
alpha, const T beta) : _alpha(alpha), _beta(beta) {}

   using output_type = C;

   static constexpr typename C::shape_type
   output_shape(const typename A::shape_type& left, const typename
B::shape_type& right) {
   }

   void operator()(const A& a, const B& b, C& result) {
      ::ten::kernels::mul_add(a, b, result, alpha, beta);
   }
};*/

/// axpy2
/// __a <- alpha * __b + __a
/*template <class __a, class __b, class __c = __a>
struct __axpy2 : ::ten::functional::func<> {
   static_assert(::ten::istensor_node<__a>::value &&
::ten::istensor_node<__b>::value
      && SameTensor<__a, __b>
      "Different tensor types.");

   using output_type = __c;

   static constexpr typename __c::shape_type
   output_shape(const typename __a::shape_type &left,
               const typename __b::shape_type &right) {
      typename __c::shape_type s(left);
      return s;
   }

   static auto output_shape(const __a &a, const __b &b) { return a.shape(); }

   void operator()(const __a &left, const __b &right, __c &result) {
      const size_type n = right.size();
      const T alpha = 1.;
      ::ten::kernels::blas::axpy(n, alpha, right.data(), 1, result.data(), 1);
   }
};*/

////////////////////////////////////////////////////////////////////////////////
// Reshape

// Reshape to static
template <class Shape> struct static_reshape {
   static_assert(Shape::is_static(), "Shape must be static");

   template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value || ::ten::is_row<A>::value) && ::ten::is_tensor<B>::value)
   struct func : ::ten::functional::func<> {
         static_assert(A::is_static(), "A must be static tensor or static column or static row.");

      using output_type = B;

      void operator()(const A &left, B &right) {
         // FIXME Fix static reshape, right now it does copy the data
         // Maybe get shape outside tensor_node
         for (size_t i = 0; i < right.size(); i++) {
            right[i] = left[i];
         }
      }
   };
};

template <size_type... __dims>
using dims_static_reshape = static_reshape<::ten::shape<__dims...>>;

// Dynamic reshape
// Shape is the target shape type
template <class Shape> struct dynamic_reshape {

   template <class A, class B>
   requires((::ten::is_tensor<A>::value || ::ten::is_column<A>::value || ::ten::is_row<A>::value) && ::ten::is_tensor<B>::value)
   struct func : ::ten::functional::func<true, true> {
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
         // FIXME Maybe move shape out of node
         // and set right = B(left.node(), _shape)
         // Copy the data
         for (size_type i = 0; i < right.size(); i++) {
            right[i] = left[i];
         }
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
      using output_type = B;

      void operator()(const A &left, B &right) {
         using shape_type = B::shape_type;
         using storage_type = B::storage_type;

         std::shared_ptr<storage_type> storage(new storage_type());
         // FIXME use auto storage =
         // std::make_shared<storage_type>(storage_type());
         right = B(storage);
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

      using output_type = B;

      typename B::shape_type output_shape(const typename A::shape_type &s) {
         typename B::shape_type res({s.dim(1), s.dim(0)});
         return res;
      }

      void operator()(const A &left, B &right) {
         using storage_type = B::storage_type;
         using shape_type = B::shape_type;

         size_type m = left.dim(0);
         size_type n = left.dim(1);
         shape_type s({left.dim(1), left.dim(0)});
         std::shared_ptr<storage_type> storage(new storage_type(s));
         // FIXME use auto storage =
         // std::make_shared<storage_type>(storage_type()); copy elements
         right = B(storage, s);
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
