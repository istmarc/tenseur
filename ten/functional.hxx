#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <type_traits>

#include <ten/kernels/binary_ops.hxx>
#include <ten/kernels/blas_api.hxx>
#include <ten/kernels/mul.hxx>
#include <ten/types.hxx>

namespace ten {
enum class binary_operation;
}

namespace ten::functional {
////////////////////////////////////////////////////////////////////////////////
// Functions types
template <bool __params = false, bool __with_shape = false> struct func {};

template <class __func> struct has_params {
   static constexpr bool value = std::is_base_of_v<func<true, true>, __func> ||
                                 std::is_base_of_v<func<true, false>, __func>;
};

template <class __func> struct has_shape {
   static constexpr bool value = std::is_base_of_v<func<true, true>, __func> ||
                                 std::is_base_of_v<func<false, true>, __func>;
};

////////////////////////////////////////////////////////////////////////////////
// TODO Apply for unary function with same input and output type
/*template<std::function<float(float)> Function> struct Apply{
   template<class A, class __b = A>
   struct fun : func<>{
      using output_type = __b;

      void operator()(const A &a, __b &b) {
         using value_type = typename __b::value_type;
         for (size_t i = 0; i < a.size(); i++) {
            b[i] = Function(static_cast<value_type>(a[i]));
         }
      }

      static typename __b::shape_type
      output_shape(const typename __b::shape_type &right) {
         return right;
      }

   };
};*/

////////////////////////////////////////////////////////////////////////////////
// Unary functions

/// Square root
template <class __a, class __b = __a> struct sqrt : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using value_type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sqrt(static_cast<value_type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Square
template <class __a, class __b = __a> struct sqr : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using value_type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = static_cast<value_type>(a[i]) * static_cast<value_type>(a[i]);
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// absolute value
template <class __a, class __b = __a> struct abs : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using value_type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::abs(static_cast<value_type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Power
template <class __a, class __b = __a> struct pow : func<true> {
 private:
   double _n;

 public:
   using output_type = __b;

   explicit pow(double n) : _n(n) {}

   void operator()(const __a &a, __b &b) const {
      using value_type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::pow(static_cast<value_type>(a[i]), _n);
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Minimum
template <class __a, class __b = typename __a::scalarnode_type>
struct min : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __a::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::min(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Maximum
template <class __a, class __b = typename __a::scalarnode_type>
struct max : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::max(static_cast<type>(a[i]), res);
      }
      b = res;
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};
/// Sum
template <class __a, class __b = typename __a::scalarnode_type>
struct sum : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      type res = 0.;
      for (size_t i = 0; i < a.size(); i++) {
         res += static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// cum__Sum
template <class __a, class __b = __a> struct cum_sum : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      b[0] = static_cast<type>(a[0]);
      for (size_t i = 1; i < a.size(); i++) {
         b[i] = static_cast<type>(a[i]) + b[i - 1];
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Prod
template <class __a, class __b = typename __a::scalarnode_type>
struct prod : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      type res = 1.;
      for (size_t i = 0; i < a.size(); i++) {
         res *= static_cast<type>(a[i]);
      }
      b = res;
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

// TODO use simd for cos, sin and tan

/// Sine
template <class __a, class __b = __a> struct sin : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sin(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// asin
template <class __a, class __b = __a> struct asin : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::asin(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Sinh
template <class __a, class __b = __a> struct sinh : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sinh(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Cosine
template <class __a, class __b = __a> struct cos : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::cos(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// acos
template <class __a, class __b = __a> struct acos : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::acos(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// cosh
template <class __a, class __b = __a> struct cosh : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::cosh(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Tangent
template <class __a, class __b = __a> struct tan : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::tan(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

template <class __a, class __b = __a> struct atan : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::atan(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

template <class __a, class __b = __a> struct tanh : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::tanh(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Exponential
template <class __a, class __b = __a> struct exp : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::exp(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Natural logarithm
template <class __a, class __b = __a> struct log : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::log(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Logarithm
template <class __a, class __b = __a> struct log10 : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::log10(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Floor
template <class __a, class __b = __a> struct floor : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::floor(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

/// Ceil
template <class __a, class __b = __a> struct ceil : func<> {
   using output_type = __b;

   void operator()(const __a &a, __b &b) {
      using type = typename __b::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::ceil(static_cast<type>(a[i]));
      }
   }

   static typename __b::shape_type
   output_shape(const typename __b::shape_type &right) {
      return right;
   }
};

////////////////////////////////////////////////////////////////////////////////
// __binary functions (__add, Sub, Mul and Div)
namespace details {
template <class, class> struct common_type;

template <TensorNode __a, TensorNode __b>
   requires same_shape<__a, __b> && same_storage_order<__a, __b> &&
            same_storage<__a, __b> && same_allocator<__a, __b>
struct common_type<__a, __b> {
   using value_type =
       std::common_type_t<typename __a::value_type, typename __b::value_type>;
   using type =
       tensor_node<value_type, typename __a::shape_type, __a::storage_order(),
                   typename __a::storage_type, typename __a::allocator_type>;
};

template <class __a, class __b>
using common_type_t = typename common_type<__a, __b>::type;
} // namespace details

// __binary function
template <::ten::binary_operation __kind> struct binary_func {

   template <class __a, class __b, class __c = details::common_type_t<__a, __b>>
   struct func : ::ten::functional::func<> {

      using output_type = __c;

      static constexpr typename __c::shape_type
      output_shape(const typename __a::shape_type &left,
                   const typename __b::shape_type &right) {
         typename __c::shape_type s(left);
         return s;
      }

      static auto output_shape(const __a &a, const __b &b) { return a.shape(); }

      void operator()(const __a &left, const __b &right, __c &result) {
         ::ten::kernels::binary_ops<__kind>(left, right, result);
      }
   };
};

namespace details {
template <class, class> struct mul_result;

// vector * vector
template <VectorNode __a, VectorNode __b>
   requires same_shape<__a, __b> && same_storage_order<__a, __b> &&
            same_storage<__a, __b> && same_allocator<__a, __b>
struct mul_result<__a, __b> {
   using value_type =
       std::common_type_t<typename __a::value_type, typename __b::value_type>;
   using type =
       tensor_node<value_type, typename __a::shape_type, __a::storage_order(),
                   typename __a::storage_type, typename __a::allocator_type>;
};

// matrix * matrix
template <MatrixNode __a, MatrixNode __b>
   requires same_storage_order<__a, __b> && same_storage<__a, __b> &&
            same_allocator<__a, __b>
struct mul_result<__a, __b> {
   using value_type =
       std::common_type_t<typename __a::value_type, typename __b::value_type>;
   using type =
       tensor_node<value_type,
                   ::ten::shape<__a::shape_type::template static_dim<0>(),
                                __b::shape_type::template static_dim<1>()>,
                   __a::storage_order(), typename __a::storage_type,
                   typename __a::allocator_type>;
};

// matrix * vector
template <MatrixNode __a, VectorNode __b>
   requires same_storage_order<__a, __b> && same_storage<__a, __b> &&
            same_allocator<__a, __b>
struct mul_result<__a, __b> {
   using value_type =
       std::common_type_t<typename __a::value_type, typename __b::value_type>;
   using type =
       tensor_node<value_type, shape<__a::shape_type::template static_dim<0>()>,
                   __a::storage_order(), typename __a::storage_type,
                   typename __a::allocator_type>;
};

// scalar * tensor
template <ScalarNode __a, TensorNode __b> struct mul_result<__a, __b> {
   using type = __b;
};

/// scalar * node
template <ScalarNode __a, Node __b> struct mul_result<__a, __b> {
   using type = std::remove_cvref_t<__b>::evaluated_type::node_type;
};

/// node * vector
template <Node __a, VectorNode __b> struct mul_result<__a, __b> {
   using evaluated_type = std::remove_cvref_t<__a>::evaluated_type::node_type;
   using value_type = mul_result<evaluated_type, __b>::value_type;
   using type = mul_result<evaluated_type, __b>::type;
};

/// matrix_node * matrix
template <Node __a, MatrixNode __b>
   requires(
       ten::is_matrix_node<
           typename std::remove_cvref_t<__a>::evaluated_type::node_type>::value)
struct mul_result<__a, __b> {
   using evaluated_type = std::remove_cvref_t<__a>::evaluated_type::node_type;
   using value_type = mul_result<evaluated_type, __b>::value_type;
   using type = mul_result<evaluated_type, __b>::type;
};

/// scalar_node * tensor
template <Node __a, TensorNode __b>
   requires(
       ten::is_scalar_node<
           typename std::remove_cvref_t<__a>::evaluated_type::node_type>::value)
struct mul_result<__a, __b> {
   // using evaluated_type =
   // std::remove_cvref_t<__a>::evaluated_type::node_type; using value_type =
   // mul_result<evaluated_type, __b>::value_type;
   using type = __b; // mul_result<evaluated_type, __b>::type;
};

/// vector * node
template <VectorNode __a, Node __b> struct mul_result<__a, __b> {
   using evaluated_type = std::remove_cvref_t<__b>::evaluated_type::node_type;
   using value_type = mul_result<evaluated_type, __a>::value_type;
   using type = mul_result<evaluated_type, __a>::type;
};

/// matrix * node
template <MatrixNode __a, Node __b> struct mul_result<__a, __b> {
   using evaluated_type = std::remove_cvref_t<__b>::evaluated_type::node_type;
   using value_type = mul_result<evaluated_type, __a>::value_type;
   using type = mul_result<evaluated_type, __a>::type;
};

/// tensor * node
template <TensorNode __a, Node __b> struct mul_result<__a, __b> {
   using evaluated_type = std::remove_cvref_t<__b>::evaluated_type::node_type;
   using value_type = mul_result<evaluated_type, __a>::value_type;
   using type = mul_result<evaluated_type, __a>::type;
};

} // namespace details

template <class __a, class __b,
          class __c = typename details::mul_result<__a, __b>::type>
struct mul;

// vector * vector
template <VectorNode __x, VectorNode __y, VectorNode __z>
struct mul<__x, __y, __z> {

   template <VectorNode __a, VectorNode __b,
             VectorNode __c = typename details::mul_result<__a, __b>::type>
   using func = binary_func<::ten::binary_operation::mul>::func<__a, __b, __c>;
};

// matrix * matrix
template <MatrixNode __x, MatrixNode __y, MatrixNode __z>
struct mul<__x, __y, __z> {

   template <MatrixNode __a, MatrixNode __b,
             MatrixNode __c = typename details::mul_result<__a, __b>::type>
   struct func : ::ten::functional::func<> {
      using output_type = __c;

      static typename __c::shape_type
      output_shape(const typename __a::shape_type &left,
                   const typename __b::shape_type &right) {
         std::initializer_list<size_type> &&dims = {left.dim(0), right.dim(1)};
         typename __c::shape_type s(std::move(dims));
         return s;
      }

      void operator()(const __a &left, const __b &right, __c &result) {
         kernels::mul(left, right, result);
      }
   };
};

// matrix * vector
template <MatrixNode __x, VectorNode __y, VectorNode __z>
struct mul<__x, __y, __z> {

   template <MatrixNode __a, VectorNode __b,
             VectorNode __c = typename details::mul_result<__a, __b>::type>
   struct func : ::ten::functional::func<> {
      using output_type = __c;

      static typename __c::shape_type
      output_shape(const typename __a::shape_type &left,
                   const typename __b::shape_type &right) {
         std::initializer_list<size_type> &&dims = {left.dim(0)};
         typename __c::shape_type s(std::move(dims));
         return s;
      }

      void operator()(const __a &left, const __b &right, __c &result) {
         kernels::mul(left, right, result);
      }
   };
};

// scalar * tensor
template <ScalarNode __x, TensorNode __y, TensorNode __z>
struct mul<__x, __y, __z> {

   template <ScalarNode __a, TensorNode __b,
             TensorNode __c = typename details::mul_result<__a, __b>::type>
   struct func : ::ten::functional::func<> {
      using output_type = __c;

      static typename __c::shape_type
      output_shape(const typename __b::shape_type &right) {
         return right;
      }

      void operator()(const __a &left, const __b &right, __c &result) {
         size_t n = result.size();
         using value_type = typename __c::value_type;
         for (size_t i = 0; i < n; i++) {
            result[i] = static_cast<value_type>(left.value()) *
                        static_cast<value_type>(right[i]);
         }
      }
   };
};

// matrix_node * matrix
template <Node __x, MatrixNode __y, MatrixNode __z>
   requires(
       ten::is_matrix_node<
           typename std::remove_cvref_t<__x>::evaluated_type::node_type>::value)
struct mul<__x, __y, __z> {
   using evaluated_type = std::remove_cvref_t<__x>::evaluated_type::node_type;
   // matrix_node * matrix_node
   template <MatrixNode __a, MatrixNode __b,
             MatrixNode __c = typename details::mul_result<__a, __b>::type>
   using func = mul<__a, __b, __c>::template func<__a, __b, __c>;
};

// scalar_node * tensor
template <Node __x, TensorNode __y, TensorNode __z>
   requires(
       ten::is_scalar_node<
           typename std::remove_cvref_t<__x>::evaluated_type::node_type>::value)
struct mul<__x, __y, __z> {
   using evaluated_type = std::remove_cvref_t<__x>::evaluated_type::node_type;
   template <ScalarNode __a, TensorNode __b,
             TensorNode __c = typename details::mul_result<__a, __b>::type>
   using func = mul<__a, __b, __c>::template func<__a, __b, __c>;
};

////////////////////////////////////////////////////////////////////////////////
/// Other __bL__aS functions

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

namespace details {
// dynamic reshape result
template <class __a, class __shape> struct reshape_result {
   using type =
       tensor_node<typename __a::value_type, __shape, __a::storage_order(),
                   typename __a::storage_type, typename __a::allocator_type>;
};
} // namespace details

// Reshape to static
template <class __shape> struct static_reshape {
   static_assert(__shape::is_static(), "Shape must be static");

   template <class __a,
             class __b = typename details::reshape_result<__a, __shape>::type>
   struct func : ::ten::functional::func<> {
      using output_type = __b;

      void operator()(const __a &left, __b &right) {
         right = __b(left.storage());
      }
   };
};

template <size_type... __dims>
using dims_static_reshape = static_reshape<::ten::shape<__dims...>>;

// Dynamic reshape
// __shape is the target shape type
template <class __shape> struct dynamic_reshape {

   template <class __a,
             class __b = typename details::reshape_result<__a, __shape>::type>
   struct func : ::ten::functional::func<true, true> {
    public:
      using output_type = __b;

    private:
      __shape _shape;

    public:
      func(const __shape &s) : _shape(s) {}
      func(__shape &&s) : _shape(std::move(s)) {}

      typename __b::shape_type output_shape(const typename __a::shape_type &) {
         return _shape;
      }

      void operator()(const __a &left, __b &right) {
         right = __b(left.storage(), _shape);
      }
   };
};

////////////////////////////////////////////////////////////////////////////////
// Transpose a matrix

namespace details {
// static transpose result
template <class __a, class __shape> struct static_transpose_result {
   static_assert(__shape::isStatic(), "Shape must be static.");
   static_assert(__shape::rank() == 2, "Shape rank must be 2.");

   using type = tensor_node<typename __a::value_type,
                            ::ten::shape<__shape::template static_dim<1>(),
                                         __shape::template static_dim<0>()>,
                            __a::storage_order(), typename __a::storage_type,
                            typename __a::allocator_type>;
};
// dynamic transpose result
template <class __a, class __shape> struct transpose_result {
   using type =
       tensor_node<typename __a::value_type, __shape, __a::storage_order(),
                   typename __a::storage_type, typename __a::allocator_type>;
};
} // namespace details

// Reshape to static
template <class __shape> struct static_transpose {
   static_assert(__shape::is_static(), "Shape must be static.");
   static_assert(__shape::rank() == 2, "Shape rank must be 2.");

   template <class __a,
             class __b =
                 typename details::static_transpose_result<__a, __shape>::type>
   struct func : ::ten::functional::func<> {
      using output_type = __b;

      void operator()(const __a &left, __b &right) {
         using shape_type = __b::shape_type;
         using storage_type = __b::storage_type;

         std::shared_ptr<storage_type> storage(new storage_type());
         // FIXME use auto storage =
         // std::make_shared<storage_type>(storage_type());
         right = __b(storage);
         // copy transposed data
         size_type m = __a::shape_type::template static_dim<0>();
         size_type n = __a::shape_type::template static_dim<1>();
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
template <class __shape> struct dynamic_transpose {

   template <class __a,
             class __b = typename details::transpose_result<__a, __shape>::type>
   struct func : ::ten::functional::func<true> {
      static_assert(__shape::is_dynamic(), "Shape must be dynamic");
      // FIXME Currently defined only for matrices
      static_assert(__shape::rank() == 2, "Shape rank must be 2.");

      using output_type = __b;

      typename __b::shape_type output_shape(const typename __a::shape_type &s) {
         typename __b::shape_type res({s.dim(1), s.dim(0)});
         return res;
      }

      void operator()(const __a &left, __b &right) {
         using storage_type = __b::storage_type;
         using shape_type = __b::shape_type;

         size_type m = left.dim(0);
         size_type n = left.dim(1);
         shape_type s({left.dim(1), left.dim(0)});
         std::shared_ptr<storage_type> storage(new storage_type(s));
         // FIXME use auto storage =
         // std::make_shared<storage_type>(storage_type()); copy elements
         right = __b(storage, s);
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
