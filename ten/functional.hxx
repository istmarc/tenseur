#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <initializer_list>
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
template <Scalar A, Scalar B> struct mul_result<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type = ::ten::scalar<value_type>;
};

// vector * vector
template <Vector A, Vector B>
   requires(A::is_dynamic() && B::is_dynamic() && same_shape<A, B> &&
            same_storage_order<A, B> && same_storage<A, B> &&
            same_allocator<A, B>)
struct mul_result<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type, typename A::shape_type, A::storage_order(),
                     typename A::storage_type, typename A::allocator_type>;
};

// svector * svector
template <StaticVector A, StaticVector B>
   requires(A::is_static() && B::is_static() && same_shape<A, B> &&
            same_storage_order<A, B> && same_storage<A, B> &&
            same_allocator<A, B>)
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

// smatrix * smatrix
template <StaticMatrix A, StaticMatrix B>
   requires(same_storage_order<A, B> && same_allocator<A, B>)
struct mul_result<A, B> {
   static_assert(A::shape_type::template static_dim<1>() ==
                     B::shape_type::template static_dim<0>(),
                 "ten::functional::mul_resul incompatible matrice shapes.\n");
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;

   using shape_type = ::ten::shape<A::shape_type::template static_dim<0>(),
                                   B::shape_type::template static_dim<1>()>;
   using type =
       ranked_tensor<value_type, shape_type, A::storage_order(),
                     ten::default_storage<value_type, shape_type>, void>;
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

// smatrix * svector
template <class A, class B>
   requires(::ten::is_smatrix_v<A> && ::ten::is_svector_v<B> &&
            same_storage_order<A, B> && same_allocator<A, B>)
struct mul_result<A, B> {
   static_assert(A::shape_type::template static_dim<1>() ==
                     B::shape_type::template static_dim<0>(),
                 "ten::functional::mul_result Incompatible static matrix and "
                 "static vector sizes.");

   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       ranked_tensor<value_type, shape<A::shape_type::template static_dim<0>()>,
                     A::storage_order(), typename A::storage_type,
                     typename A::allocator_type>;
};

// scalar * tensor
// template <Scalar A, Tensor B> struct mul_result<A, B> {
//   using type = B;
//};

// tensor * tensor
template <Tensor X, Tensor Y>
   requires(X::order() >= 3 && Y::order() >= 3 && same_shape<X, Y> &&
            same_storage_order<X, Y> && same_storage<X, Y> &&
            same_allocator<X, Y>)
struct mul_result<X, Y> {
   using value_type =
       std::common_type_t<typename X::value_type, typename Y::value_type>;
   using type =
       ranked_tensor<value_type, typename X::shape_type, X::storage_order(),
                     typename X::storage_type, typename X::allocator_type>;
};

// stensor * stensor
template <StaticTensor X, StaticTensor Y>
   requires(X::order() >= 3 && Y::order() >= 3 && same_shape<X, Y> &&
            same_storage_order<X, Y> && same_storage<X, Y> &&
            same_allocator<X, Y>)
struct mul_result<X, Y> {
   using value_type =
       std::common_type_t<typename X::value_type, typename Y::value_type>;
   using type =
       ranked_tensor<value_type, typename X::shape_type, X::storage_order(),
                     typename X::storage_type, typename X::allocator_type>;
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

/// UnaryExpr * svector
template <UnaryExpr A, StaticVector B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * vector
template <BinaryExpr A, Vector B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * svector
template <BinaryExpr A, StaticVector B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// UnaryExpr * matrix
template <UnaryExpr A, Matrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// UnaryExpr * smatrix
template <UnaryExpr A, StaticMatrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// matrix * UnaryExpr
template <Matrix A, UnaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// smatrix * UnaryExpr
template <StaticMatrix A, UnaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// BinaryExpr * matrix
template <BinaryExpr A, Matrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * smatrix
template <BinaryExpr A, StaticMatrix B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// Matrix * BinaryExpr
template <Matrix A, BinaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// Smatrix * BinaryExpr
template <StaticMatrix A, BinaryExpr B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<B>::output_type;
   using type = mul_result<evaluated_type, A>::type;
};

/// UnaryExpr * tensor
template <UnaryExpr A, Tensor B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// UnaryExpr * stensor
template <UnaryExpr A, StaticTensor B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * tensor
template <BinaryExpr A, Tensor B> struct mul_result<A, B> {
   using evaluated_type = std::remove_cvref_t<A>::output_type;
   using type = mul_result<evaluated_type, B>::type;
};

/// BinaryExpr * stensor
template <BinaryExpr A, StaticTensor B> struct mul_result<A, B> {
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
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqrt : func<> {
   static constexpr std::string name() { return std::string("sqrt"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y.value() = std::sqrt(x.value());
      }
      if constexpr (::ten::is_tensor_v<X> || ::ten::is_column_v<X> ||
                    ::ten::is_row_v<X>) {
         if (!y) {
            if constexpr (Y::is_dynamic()) {
               y = std::make_shared<Y>(x->shape());
            } else {
               y = std::make_shared<Y>();
            }
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] =
                std::sqrt(static_cast<output_value_type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::sqrt gradient incompatible shapes\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = 1 / (2 * std::sqrt(x[i]));
         }
      }
   }
};

/// Square
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqr : func<> {
   static constexpr std::string name() { return std::string("sqr"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (::ten::is_scalar<X>::value) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y.value() = x.value() * x.value();
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         auto xarr = *x.get();
         auto yarr = *y.get();
         ::ten::kernels::binary_ops<::ten::binary_operation::mul>(xarr, xarr,
                                                                  yarr);
         /*
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = static_cast<output_value_type>(x[i]) *
                   static_cast<output_value_type>(x[i]);
         }*/
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad->value() = 2 * x->value();
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::sqr gradient incompatible shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = 2 * x[i];
         }
      }
   }
};

/// absolute value
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct abs : func<> {
   static constexpr std::string name() { return std::string("abs"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y.value() = std::abs(x.value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::abs(static_cast<value_type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.value() >= 0) {
         grad.value() = 1;
      } else {
         grad.value() = -1;
      }
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::abs gradient incompatible shape\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            if (x[i] >= 0) {
               grad[i] = 1;
            } else {
               grad[i] = -1;
            }
         }
      }
   }
};

/// Power
template <class X, class Y>
   requires(((ten::is_tensor<X>::value || ten::is_column<X>::value ||
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) const {
      using value_type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y.value() = std::pow(x.value(), _n);
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x.size(); i++) {
            y[i] = std::pow(static_cast<value_type>(x[i]), _n);
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() =
          static_cast<output_value_type>(_n) * std::pow(x.value(), _n - 1);
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::pow gradient incompatible shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] =
                static_cast<output_value_type>(_n) * std::pow(x[i], _n - 1);
         }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>();
      }
      using type = typename X::value_type;
      type res = (*x.get())[0];
      for (size_t i = 1; i < x->size(); i++) {
         res = std::min((*x.get())[i], res);
      }
      y->value() = res;
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>();
      }
      using type = typename Y::value_type;
      type res = (*x.get())[0];
      for (size_t i = 1; i < x->size(); i++) {
         res = std::max(static_cast<type>((*x.get())[i]), res);
      }
      y->value() = res;
   }
};

/// Mean
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_scalar<Y>::value)
struct mean : func<> {
   static constexpr std::string name() { return std::string("mean"); }

   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>();
      }
      using type = typename Y::value_type;
      type res = 0.;
      for (size_t i = 0; i < x->size(); i++) {
         res += static_cast<type>((*x.get())[i]);
      }
      y->value() = res / x.size();
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      using T = X::value_type;
      for (size_t i = 0; i < x.size(); i++) {
         grad[i] = T(1) / T(x.size());
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>();
      }
      using type = typename Y::value_type;
      type res = 0.;
      for (size_t i = 0; i < x->size(); i++) {
         res += static_cast<type>((*x.get())[i]);
      }
      y->value() = res;
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::fucntional::sum gradient incompatible shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = 1;
         }
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      auto yarr = *(y.get());
      auto xarr = *(x.get());
      yarr[0] = static_cast<type>(xarr[0]);
      for (size_t i = 1; i < x->size(); i++) {
         yarr[i] = static_cast<type>(xarr[i]) + yarr[i - 1];
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>();
      }
      using type = typename Y::value_type;
      type res = 1.;
      for (size_t i = 0; i < x->size(); i++) {
         res *= static_cast<type>((*x.get())[i]);
      }
      y->value() = res;
   }
};

/// Sine
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sin : func<> {
   static constexpr std::string name() { return std::string("sin"); }

   using output_type = Y;
   using value_type = X::value_type;
   using output_value_type = Y::value_type;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::sin(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::sin(static_cast<type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = std::cos(x.value());
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::sin gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = std::cos(x[i]);
         }
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      for (size_t i = 0; i < x->size(); i++) {
         (*y.get())[i] = std::asin(static_cast<type>((*x.get())[i]));
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      for (size_t i = 0; i < x->size(); i++) {
         (*y.get())[i] = std::sinh(static_cast<type>((*x.get())[i]));
      }
   }
};

/// Cosine
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct cos : func<> {
   static constexpr std::string name() { return std::string("cos"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::cos(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] =
                std::cos(static_cast<output_value_type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = -std::sin(x.value());
   }

   /*
   template <class Gradient>
      requires(::ten::is_scalar_v<X>)
   void gradient(std::shared_ptr<X> &x, std::shared_ptr<Gradient> &y) {
      for (size_t i = 0; i < y.size(); i++) {
         y[i] = -std::sin(x.value());
      }
   }*/

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::cos gradient different sizes\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = -std::sin(x[i]);
         }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      for (size_t i = 0; i < x->size(); i++) {
         (*y.get())[i] = std::acos(static_cast<type>((*x.get())[i]));
      }
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

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      for (size_t i = 0; i < x->size(); i++) {
         (*y.get())[i] = std::cosh(static_cast<type>((*x.get())[i]));
      }
   }
};

/// Tangent
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tan : func<> {
   static constexpr std::string name() { return std::string("tan"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (::ten::is_scalar_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::tan(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] =
                std::tan(static_cast<output_value_type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      value_type z = std::tan(x.value());
      grad.value() = 1 + z * z;
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::tan gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            value_type t = std::tan(x[i]);
            grad[i] = 1 + t * t;
         }
      }
   }
};

template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct atan : func<> {
   static constexpr std::string name() { return std::string("atan"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::atan(static_cast<type>(x->value()));
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::atan(static_cast<type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = 1 / (1 + x.value() * x.value());
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::atan gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x->size(); i++) {
            grad[i] = 1 / (1 + x[i] * x[i]);
         }
      }
   }
};

template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tanh : func<> {
   static constexpr std::string name() { return std::string("tanh"); }

   using value_type = X::value_type;
   using output_value_type = Y::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y.value() = std::tanh(x.value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::tanh(static_cast<type>((*x.get())[i]));
         }
      }
   }
};

/// Exponential
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct exp : func<> {
   static constexpr std::string name() { return std::string("exp"); }

   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::exp(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::exp(static_cast<type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = std::exp(x.value());
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::exp gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            grad[i] = std::exp(x[i]);
         }
      }
   }
};

/// Natural logarithm
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct log : func<> {
   static constexpr std::string name() { return std::string("log"); }

   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::log(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::log(static_cast<type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = value_type(1) / x.value();
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::log gradient different sizes.\n";
      } else {
         for (size_t i = 0; i < x->size(); i++) {
            grad[i] = value_type(1) / x[i];
         }
      }
   }
};

/// Logarithm
template <class X, class Y>
   requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
              ::ten::is_row<X>::value) &&
             ::ten::is_tensor<Y>::value) ||
            (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct log10 : func<> {
   static constexpr std::string name() { return std::string("log10"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      using type = typename Y::value_type;
      if constexpr (::ten::is_scalar_v<X>) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         y->value() = std::log10(x->value());
      }
      if constexpr (::ten::is_tensor_v<Y>) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::log10(static_cast<type>((*x.get())[i]));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      grad.value() = value_type(1) / (x.value() * std::log(value_type(10)));
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::log10 gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x->size(); i++) {
            grad[i] = value_type(1) / (x[i] * std::log(value_type(10)));
         }
      }
   }
};

/// Floor
template <class X, class Y>
   requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value)
struct floor : func<> {
   static constexpr std::string name() { return std::string("floor"); }

   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if (!y) {
         y = std::make_shared<Y>(x->shape());
      }
      using type = typename Y::value_type;
      for (size_t i = 0; i < x->size(); i++) {
         (*y.get())[i] = std::floor(static_cast<type>((*x.get())[i]));
      }
   }
};

/// Ceil
template <class X, class Y>
   requires(((::ten::is_tensor_v<X> || ::ten::is_column_v<X> ||
              ::ten::is_row_v<X>) &&
             ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct ceil : func<> {
   static constexpr std::string name() { return std::string("ceil"); }

   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (Y::is_dynamic()) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         using type = typename Y::value_type;
         for (size_t i = 0; i < x->size(); i++) {
            (*y.get())[i] = std::ceil(static_cast<type>((*x.get())[i]));
         }
      } else {
         if (!y) {
            y = std::make_shared<Y>();
         }
         using type = typename Y::value_type;
         for (size_t i = 0; i < X::static_size(); i++) {
            (*y.get())[i] = std::ceil(static_cast<type>((*x.get())[i]));
         }
      }
   }
};

////////////////////////////////////////////////////////////////////////////////
// Binary functions (Add, Sub, Mul and Div)

/// Binary function
template <::ten::binary_operation Kind> struct binary_func {

   template <class X, class Y, class Z>
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

      using output_type = Z;

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                      std::shared_ptr<Z> &z) {
         if constexpr (Z::is_dynamic()) {
            if (x->shape() != y->shape()) {
               std::cerr << "ten::functional::binary_func<" << name()
                         << ">, input have different shapes\n";
            } else {
               if (!z) {
                  z = std::make_shared<Z>(x->shape());
               }
               auto xarr = *x.get();
               auto yarr = *y.get();
               auto zarr = *z.get();
               ::ten::kernels::binary_ops<Kind>(xarr, yarr, zarr);
            }
         } else {
            if (!std::is_same_v<typename X::shape_type,
                                typename Y::shape_type>) {
               std::cerr << "ten::functional::binary_func<" << name()
                         << ">, input have different shapes\n";
            } else {
               if (!z) {
                  z = std::make_shared<Z>();
               }
               ::ten::kernels::binary_ops<Kind>(*y.get(), *y.get(), *z.get());
            }
         }
      }

      // Compute dz(x,y)/dx in grad
      template <class Gradient>
         requires(::ten::is_tensor_v<Gradient>)
      void gradient_left(const X & /*x*/, const Y &y, Gradient &grad) {
         if (y.shape() != grad.shape()) {
            std::cerr << "ten::functional::binary_func<" << name()
                      << "> gradient ";
            std::cerr << "different shapes.\n";
         } else {
            if constexpr ((Kind == ::ten::binary_operation::add) ||
                          (Kind == ::ten::binary_operation::sub)) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = 1;
               }
            }
            if (Kind == ::ten::binary_operation::mul) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = y[i];
               }
            }
            if (Kind == ::ten::binary_operation::div) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = 1 / y[i];
               }
            }
         }
      }

      template <class Gradient>
         requires(::ten::is_stensor_v<Gradient>)
      void gradient_left(const X & /*x*/, const Y &y, Gradient &grad) {
         if constexpr (!std::is_same_v<typename Y::shape_type,
                                       typename Gradient::shape_type>) {
            std::cerr << "ten::functional::binary_func<" << name()
                      << "> gradient ";
            std::cerr << "different shapes.\n";
         } else {
            if constexpr ((Kind == ::ten::binary_operation::add) ||
                          (Kind == ::ten::binary_operation::sub)) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = 1;
               }
            }
            if (Kind == ::ten::binary_operation::mul) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = y[i];
               }
            }
            if (Kind == ::ten::binary_operation::div) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = 1 / y[i];
               }
            }
         }
      }

      // Compute dz(x,y)/dy in grad
      template <class Gradient>
         requires(::ten::is_tensor_v<Gradient>)
      void gradient_right(const X &x, const Y &y, Gradient &grad) {
         if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
            std::cerr << "ten::functional::binary_func<" << name()
                      << "> gradient ";
            std::cerr << "different shapes.\n";
         } else {
            if constexpr (Kind == ::ten::binary_operation::add) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = 1;
               }
            }
            if constexpr (Kind == ::ten::binary_operation::sub) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = -1;
               }
            }
            if (Kind == ::ten::binary_operation::mul) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = x[i];
               }
            }
            if (Kind == ::ten::binary_operation::div) {
               for (size_t i = 0; i < grad.size(); i++) {
                  grad[i] = -x[i] / (y[i] * y[i]);
               }
            }
         }
      }

      template <class Gradient>
         requires(::ten::is_stensor_v<Gradient>)
      void gradient_right(const X &x, const Y &y, Gradient &grad) {
         if (!std::is_same_v<typename X::shape_type,
                             typename Gradient::shape_type> ||
             !std::is_same_v<typename Y::shape_type,
                             typename Gradient::shape_type>) {
            std::cerr << "ten::functional::binary_func<" << name()
                      << "> gradient ";
            std::cerr << "different shapes.\n";
         } else {
            if constexpr (Kind == ::ten::binary_operation::add) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = 1;
               }
            }
            if constexpr (Kind == ::ten::binary_operation::sub) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = -1;
               }
            }
            if (Kind == ::ten::binary_operation::mul) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = x[i];
               }
            }
            if (Kind == ::ten::binary_operation::div) {
               for (size_t i = 0; i < Gradient::static_size(); i++) {
                  grad[i] = -x[i] / (y[i] * y[i]);
               }
            }
         }
      }
   };
};

/// Binary function with scalar
template <::ten::binary_operation Kind> struct scalar_left_binary_func {

   template <Scalar X, class Y, class Z>
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

      using output_type = Z;

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                      std::shared_ptr<Z> &z) {
         if constexpr (Z::is_dynamic()) {
            if (!z) {
               z = std::make_shared<Z>(y->shape());
            }
            Y xarr(y->shape());
            for (size_t i = 0; i < y->size(); i++) {
               xarr[i] = x->value();
            }
            ::ten::kernels::binary_ops<Kind>(xarr, *y.get(), *z.get());
         } else {
            if (!z) {
               z = std::make_shared<Z>();
            }
            Y xarr;
            for (size_t i = 0; i < Y::static_size(); i++) {
               xarr[i] = x->value();
            }
            ::ten::kernels::binary_ops<Kind>(xarr, *y.get(), *z.get());
         }
      }
   };
};

/// Binary function with scalar right type
template <::ten::binary_operation Kind> struct scalar_right_binary_func {

   template <class X, Scalar Y, class Z>
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

      using output_type = Z;

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                      std::shared_ptr<Z> &z) {
         if constexpr (Z::is_dynamic()) {
            if (!z) {
               z = std::make_shared<Z>(x->shape());
            }
         } else {
            if (!z) {
               z = std::make_shared<Z>();
            }
         }
         if constexpr (Z::is_static()) {
            X yarr;
            for (size_t i = 0; i < x->size(); i++) {
               yarr[i] = y->value();
            }
            ::ten::kernels::binary_ops<Kind>(*x.get(), yarr, *z.get());
         } else {
            X yarr(x->shape());
            for (size_t i = 0; i < x->size(); i++) {
               yarr[i] = y->value();
            }
            ::ten::kernels::binary_ops<Kind>(*x.get(), yarr, *z.get());
         }
      }
   };
};

template <class A, class B, class C> struct mul : func<> {};

// vector * vector
template <DynamicVector X, DynamicVector Y, DynamicVector Z>
struct mul<X, Y, Z> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if (x->shape() != y->shape()) {
         std::cerr << "ten::functional::mul, different input shapes\n";
      } else {
         if (!z) {
            z = std::make_shared<Z>(x->shape());
         }
         ::ten::kernels::binary_ops<::ten::binary_operation::mul>(
             *x.get(), *y.get(), *z.get());
      }
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<Gradient>)
   void gradient_left(const X &x, const Y &y, Gradient &grad) {
      if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
         std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = y[i];
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<Gradient>)
   void gradient_right(const X &x, const Y &y, Gradient &grad) {
      if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
         std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = x[i];
         }
      }
   }
};

// svector * svector
template <StaticVector X, StaticVector Y, StaticVector Z>
struct mul<X, Y, Z> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if (!std::is_same_v<typename X::shape_type, typename Y::shape_type>) {
         std::cerr << "ten::functional::mul, different input shapes\n";
      } else {
         if (!z) {
            z = std::make_shared<Z>();
         }
         ::ten::kernels::binary_ops<::ten::binary_operation::mul>(
             *x.get(), *y.get(), *z.get());
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<Gradient>)
   void gradient_left(const X &x, const Y &y, Gradient &grad) {
      if constexpr (!std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type> ||
                    !std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            grad[i] = y[i];
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<Gradient>)
   void gradient_right(const X &x, const Y &y, Gradient &grad) {
      if constexpr (!std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type> ||
                    !std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            grad[i] = x[i];
         }
      }
   }
};

// matrix * matrix
template <Matrix X, Matrix Y, Matrix Z> struct mul<X, Y, Z> : func<> {

   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if (x->shape().dim(1) != y->shape().dim(0)) {
         std::cerr
             << "ten::functional::mul - incompatibles input matrices shapes.\n";
      } else {
         if (!z) {
            std::initializer_list<size_type> &&dims = {x->shape().dim(0),
                                                       y->shape().dim(1)};
            typename Z::shape_type zshape(std::move(dims));
            z = std::make_shared<Z>(zshape);
         }
         kernels::mul(*x.get(), *y.get(), *z.get());
      }
   }

   // Static shapes gradient
   template <class Gradient>
      requires(::ten::is_matrix_v<X> && ::ten::is_matrix_v<Y> &&
               ::ten::is_matrix_v<Gradient>)
   void gradient_left(const X &x, const Y &y, Gradient &grad) {
      if (x.shape().dim(1) != y.shape().dim(0)) {
         std::cerr << "ten::functional::mul gradient - incompatible shapes.\n";
      } else if ((x.shape().dim(0) != grad.shape().dim(0)) ||
                 (x.shape().dim(1) != grad.shape().dim(1))) {
         std::cerr
             << "ten::functional::mul gradient - incompatible output shape.\n";
      } else {
         if constexpr (Gradient::shape_type::is_dynamic()) {
            auto rows = grad.shape().dim(0);
            auto cols = grad.shape().dim(1);
            for (size_t i = 0; i < cols; i++) {
               grad(0, i) = 0;
               for (size_t j = 0; j < y.shape().dim(1); j++) {
                  grad(0, i) += y(i, j);
               }
            }
            for (size_t j = 0; j < cols; j++) {
               for (size_t i = 1; i < rows; i++) {
                  grad(i, j) = grad(0, j);
               }
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_matrix_v<X> && ::ten::is_matrix_v<Y> &&
               ::ten::is_matrix_v<Gradient>)
   void gradient_right(const X &x, const Y &y, Gradient &grad) {
      if (x.shape().dim(1) != y.shape().dim(0)) {
         std::cerr << "ten::functional::mul gradient incompatible shapes.\n";
      } else if ((y.shape().dim(0) != grad.shape().dim(0)) ||
                 (y.shape().dim(1) != grad.shape().dim(1))) {
         std::cerr
             << "ten::functional::mul gradient incompatible output shape.\n";
      } else {
         if constexpr (Gradient::shape_type::is_dynamic()) {
            auto rows = grad.shape().dim(0);
            auto cols = grad.shape().dim(1);
            size_t j = 0;
            for (size_t i = 0; i < rows; i++) {
               grad(i, 0) = 0;
               for (size_t k = 0; k < x.shape().dim(0); k++) {
                  grad(i, 0) += x(k, j);
               }
               j++;
            }
            for (size_t j = 1; j < cols; j++) {
               for (size_t i = 0; i < rows; i++) {
                  grad(i, j) = grad(i, 0);
               }
            }
         }
      }
   }
};

// smatrix * smatrix
template <StaticMatrix X, StaticMatrix Y, StaticMatrix Z>
struct mul<X, Y, Z> : func<> {

   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if constexpr (X::shape_type::template static_dim<1>() !=
                    Y::shape_type::template static_dim<0>()) {
         std::cerr
             << "ten::functional::mul - incompatibles input matrices shapes.\n";
      } else {
         if (!z) {
            z = std::make_shared<Z>();
         }
         kernels::mul(*x.get(), *y.get(), *z.get());
      }
   }

   // Compute dz(x,y)/dx in grad
   template <class Gradient>
      requires(::ten::is_smatrix_v<Gradient>)
   void gradient_left(const X & /*x*/, const Y &y, Gradient &grad) {
      if constexpr (X::shape_type::template static_dim<1>() !=
                    Y::shape_type::template static_dim<0>()) {
         std::cerr << "ten::functional::mul gradient - incompatible shapes.\n";
      } else if constexpr ((X::shape_type::template static_dim<0>() !=
                            Gradient::shape_type::template static_dim<0>()) ||
                           (X::shape_type::template static_dim<1>() !=
                            Gradient::shape_type::template static_dim<1>())) {
         std::cerr
             << "ten::functional::mul gradient - incompatible output shape.\n";
      } else {
         constexpr size_t rows = Gradient::shape_type::template static_dim<0>();
         constexpr size_t cols = Gradient::shape_type::template static_dim<1>();
         for (size_t i = 0; i < cols; i++) {
            grad(0, i) = 0;
            for (size_t j = 0; j < Y::shape_type::template static_dim<1>();
                 j++) {
               grad(0, i) += y(i, j);
            }
         }
         for (size_t j = 0; j < cols; j++) {
            for (size_t i = 1; i < rows; i++) {
               grad(i, j) = grad(0, j);
            }
         }
      }
   }

   // Compute dz(x, y)/dy in grad
   template <class Gradient>
      requires(::ten::is_matrix_v<X> && ::ten::is_matrix_v<Y> &&
               ::ten::is_matrix_v<Gradient>)
   void gradient_right(const X &x, const Y & /*y*/, Gradient &grad) {
      if constexpr (X::shape_type::template static_dim<1>() !=
                    Y::shape_type::template static_dim<0>()) {
         std::cerr << "ten::functional::mul gradient - incompatible shapes.\n";
      } else if constexpr ((Y::shape_type::template static_dim<0>() !=
                            Gradient::shape_type::template static_dim<0>()) ||
                           (Y::shape_type::template static_dim<1>() !=
                            Gradient::shape_type::template static_dim<1>())) {
         std::cerr
             << "ten::functional::mul gradient - incompatible output shape.\n";
      } else {
         constexpr size_t rows = Gradient::shape_type::template static_dim<0>();
         constexpr size_t cols = Gradient::shape_type::template static_dim<1>();
         size_t j = 0;
         for (size_t i = 0; i < rows; i++) {
            grad(i, 0) = 0;
            for (size_t k = 0; k < X::shape_type::template static_dim<0>();
                 k++) {
               grad(i, 0) += x(k, j);
            }
            j++;
         }
         for (size_t j = 1; j < cols; j++) {
            for (size_t i = 0; i < rows; i++) {
               grad(i, j) = grad(i, 0);
            }
         }
      }
   }
};

// matrix * vector
template <Matrix X, Vector Y, Vector Z>
struct mul<X, Y, Z> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if (x->shape().dim(1) != y->shape().dim(0)) {
         std::cerr << "ten::functional::mul matrix vector incompatible input "
                      "shapes\n";
      } else {
         if (!z) {
            std::initializer_list<size_type> &&dims = {x->shape().dim(0)};
            typename Z::shape_type zshape(std::move(dims));
            z = std::make_shared<Z>(zshape);
         }
         kernels::mul(*x.get(), *y.get(), *z.get());
      }
   }

   template <class Gradient>
      requires(::ten::is_matrix_v<Gradient>)
   void gradient_left(const X & /*x*/, const Y &y, Gradient &grad) {
      if constexpr (Gradient::shape_type::is_dynamic()) {
         auto rows = grad.shape().dim(0);
         auto cols = grad.shape().dim(1);
         for (size_t i = 0; i < cols; i++) {
            grad(0, i) = y[i];
         }
         for (size_t j = 0; j < cols; j++) {
            for (size_t i = 1; i < rows; i++) {
               grad(i, j) = grad(0, j);
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_vector_v<Gradient>)
   void gradient_right(const X &x, const Y & /*y*/, Gradient &grad) {
      if constexpr (Gradient::shape_type::is_dynamic()) {
         size_t j = 0;
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = 0;
            for (size_t k = 0; k < x.shape().dim(0); k++) {
               grad[i] += x(k, j);
            }
            j++;
         }
      }
   }
};

// smatrix * svector
template <StaticMatrix X, StaticVector Y, StaticVector Z>
struct mul<X, Y, Z> : ::ten::functional::func<> {
   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if constexpr (X::shape_type::template static_dim<1>() !=
                    Y::shape_type::template static_dim<0>()) {
         std::cerr << "ten::functional::mul matrix vector incompatible input "
                      "shapes\n";
      } else {
         if (!z) {
            z = std::make_shared<Z>();
         }
         kernels::mul(*x.get(), *y.get(), *z.get());
      }
   }

   template <class Gradient>
      requires(::ten::is_smatrix_v<Gradient>)
   void gradient_left(const X & /*x*/, const Y &y, Gradient &grad) {
      constexpr size_t rows = Gradient::shape_type::template static_dim<0>();
      constexpr size_t cols = Gradient::shape_type::template static_dim<1>();
      for (size_t i = 0; i < cols; i++) {
         grad(0, i) = y[i];
      }
      for (size_t j = 0; j < cols; j++) {
         for (size_t i = 1; i < rows; i++) {
            grad(i, j) = grad(0, j);
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_svector_v<Gradient>)
   void gradient_right(const X &x, const Y & /*y*/, Gradient &grad) {
      size_t j = 0;
      for (size_t i = 0; i < Gradient::static_size(); i++) {
         grad[i] = 0;
         for (size_t k = 0; k < X::shape_type::template static_dim<0>(); k++) {
            grad[i] += x(k, j);
         }
         j++;
      }
   }
};

// scalar * tensor
/*
template <Scalar X, Tensor Y, Tensor Z> struct mul<X, Y, Z> : func<> {

   static constexpr std::string name() { return std::string("mul"); }

   using output_type = Z;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if constexpr (Z::is_dynamic()) {
         if (!z) {
            z = std::make_shared<Z>(y->shape());
         }
         Y xarr(y->shape());
         for (size_t i = 0; i < xarr.size(); i++) {
            xarr[i] = x->value();
         }
         ten::kernels::mul(xarr, *y.get(), *z.get());
      } else {
         if (!z) {
            z = std::make_shared<Z>();
         }
         Y xarr;
         for (size_t i = 0; i < Y::static_size(); i++) {
            xarr[i] = x->value();
         }
         ten::kernels::mul(xarr, *y.get(), *z.get());
      }
   }
};*/

////////////////////////////////////////////////////////////////////////////////
// Reshape

// Reshape to static
template <class Shape> struct static_reshape {
   static constexpr std::string name() { return std::string("static_reshape"); }

   static_assert(Shape::is_static(), "Shape must be static");

   template <class X, class Y>
      requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
                ::ten::is_row<X>::value) &&
               ::ten::is_tensor<Y>::value)
   struct func : ::ten::functional::func<false, false> {
      static_assert(X::is_static(),
                    "X must be static tensor or static column or static row.");

      using output_type = Y;

      void operator()(std::shared_ptr<X> &left, std::shared_ptr<Y> &right) {
         // Does not copy the data
         right = std::make_shared<Y>(Y(left->node(), left->format()));
      }
   };
};

template <size_type... __dims>
using dims_static_reshape = static_reshape<::ten::shape<__dims...>>;

// Dynamic reshape
// Shape is the target shape type
template <class Shape> struct dynamic_reshape {

   template <class X, class Y>
      requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
                ::ten::is_row<X>::value) &&
               ::ten::is_tensor<Y>::value)
   struct func : ::ten::functional::func<true, true> {
      static constexpr std::string name() { return std::string("reshape"); }

    public:
      using output_type = Y;

    private:
      Shape _shape;

    public:
      func(const Shape &s) : _shape(s) {}
      func(Shape &&s) : _shape(std::move(s)) {}

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
         // Does not copy the data
         y = std::make_shared<Y>(x->node(), _shape, x->format());
      }
   };
};

////////////////////////////////////////////////////////////////////////////////
// Transpose a matrix

// Reshape to static
template <class Shape> struct static_transpose {
   static_assert(Shape::is_static(), "Shape must be static.");
   static_assert(Shape::rank() == 2, "Shape rank must be 2.");

   template <class X, class Y> struct func : ::ten::functional::func<> {
      static constexpr std::string name() {
         return std::string("static_transpose");
      }

      using output_type = Y;

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
         if (!y) {
            y = std::make_shared<Y>();
         }
         using shape_type = Y::shape_type;
         // copy transposed data
         size_type m = X::shape_type::template static_dim<0>();
         size_type n = X::shape_type::template static_dim<1>();
         for (size_type i = 0; i < n; i++) {
            for (size_type j = 0; j < m; j++) {
               (*y.get())(i, j) = (*x.get())(j, i);
            }
         }
      }
   };
};

// Dynamic transpose
template <class Shape> struct dynamic_transpose {

   template <class X, class Y> struct func : ::ten::functional::func<true> {
      static_assert(Shape::is_dynamic(), "Shape must be dynamic");
      // FIXME Currently defined only for matrices
      static_assert(Shape::rank() == 2, "Shape rank must be 2.");

      static constexpr std::string name() { return std::string("transpose"); }

      using output_type = Y;

      void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
         if (!y) {
            typename Y::shape_type yshape(
                {x->shape().dim(1), x->shape().dim(0)});
            y = std::make_shared<Y>(yshape);
         }
         size_type m = x->shape().dim(0);
         size_type n = x->shape().dim(1);
         for (size_type i = 0; i < n; i++) {
            for (size_type j = 0; j < m; j++) {
               (*y.get())(i, j) = (*x.get())(j, i);
            }
         }
      }
   };
};

////////////////////////////////////////////////////////////////////////////////
// Neural networks activation functions

/// Relu
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct relu : func<> {
   static constexpr std::string name() { return std::string("relu"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (Y::is_dynamic()) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            std::cerr << "ten::functional::relu different sizes" << std::endl;
         } else {
            auto xarr = *x.get();
            auto yarr = *y.get();
            for (size_t i = 0; i < y->size(); i++) {
               if (xarr[i] >= 0) {
                  yarr[i] = xarr[i];
               } else {
                  yarr[i] = 0;
               }
            }
         }
      } else {
         if (!y) {
            y = std::make_shared<Y>();
         }
         if constexpr (X::static_size() != Y::static_size()) {
            std::cerr << "ten::functional::relu different sizes" << std::endl;
         } else {
            auto xarr = *x.get();
            auto yarr = *y.get();
            for (size_t i = 0; i < Y::static_size(); i++) {
               if (xarr[i] >= 0) {
                  yarr[i] = xarr[i];
               } else {
                  yarr[i] = 0;
               }
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < grad->size(); i++) {
            if (xarr[i] >= 0) {
               gradarr[i] = 1;
            } else {
               gradarr[i] = 0;
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            if (xarr[i] >= 0) {
               gradarr[i] = 1;
            } else {
               gradarr[i] = 0;
            }
         }
      }
   }
};

/// Leaky relu
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct leaky_relu : func<> {
   static constexpr std::string name() { return std::string("leaky_relu"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (Y::is_dynamic()) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            std::cerr << "ten::functional::relu different sizes" << std::endl;
         } else {
            auto xarr = *x.get();
            auto yarr = *y.get();
            for (size_t i = 0; i < y->size(); i++) {
               if (xarr[i] <= 0) {
                  yarr[i] = value_type(0.01) * xarr[i];
               } else {
                  yarr[i] = xarr[i];
               }
            }
         }
      } else {
         if (!y) {
            y = std::make_shared<Y>();
         }
         if constexpr (X::static_size() != Y::static_size()) {
            std::cerr << "ten::functional::relu different sizes" << std::endl;
         } else {
            auto xarr = *x.get();
            auto yarr = *y.get();
            for (size_t i = 0; i < Y::static_size(); i++) {
               if (xarr[i] <= 0) {
                  yarr[i] = value_type(0.01) * xarr[i];
               } else {
                  yarr[i] = xarr[i];
               }
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr
             << "ten::functional::leaky_relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < grad->size(); i++) {
            if (xarr[i] < 0) {
               gradarr[i] = 0.01;
            } else {
               gradarr[i] = 1;
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         std::cerr
             << "ten::functional::leaky_relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            if (xarr[i] < 0) {
               gradarr[i] = 0.01;
            } else {
               gradarr[i] = 1;
            }
         }
      }
   }
};

/// Sigmoid
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct sigmoid : func<> {
   static constexpr std::string name() { return std::string("sigmoid"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
      if constexpr (X::is_dynamic()) {
         if (!y) {
            y = std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            std::cerr << "ten::functional::relu different input sizes\n";
         } else {
            for (size_t i = 0; i < y->size(); i++) {
               (*y.get())[i] =
                   value_type(1) / (value_type(1) + std::exp(-(*x.get())[i]));
            }
         }
      } else {
         if (!y) {
            y = std::make_shared<Y>();
         }
         if (X::static_size() != Y::static_size()) {
            std::cerr << "ten::functional::relu different input sizes\n";
         } else {
            for (size_t i = 0; i < Y::static_size(); i++) {
               (*y.get())[i] =
                   value_type(1) / (value_type(1) + std::exp(-(*x.get())[i]));
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            value_type e = std::exp(-x[i]);
            grad[i] = e / ((1 + e) * (1 + e));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < X::static_size(); i++) {
            value_type e = std::exp(-x[i]);
            grad[i] = e / ((1 + e) * (1 + e));
         }
      }
   }
};

/// MSE Loss
template <class X, class Y, class Z>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y> &&
             ::ten::is_scalar_v<Z>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y> &&
             ::ten::is_scalar_v<Z>))
struct mse : func<> {
   static constexpr std::string name() { return std::string("mse"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(std::shared_ptr<X> &x, std::shared_ptr<Y> &y,
                   std::shared_ptr<Z> &z) {
      if constexpr (X::is_dynamic() && Y::is_dynamic()) {
         if (x->size() != y->size()) {
            std::cerr << "ten::functional::mse different input sizes.\n";
         } else {
            if (!z) {
               z = std::make_shared<Z>();
            }
            value_type res = 0;
            for (size_t i = 0; i < x->size(); i++) {
               value_type diff = (*x.get())[i] - (*y.get())[i];
               res += diff * diff;
            }
            z->value() = res / x->size();
         }
      } else {
         if constexpr (X::static_size() != Y::static_size()) {
            std::cerr << "ten::functional::mse different input sizes.\n";
         } else {
            if (!z) {
               z = std::make_shared<Z>();
            }
            value_type res = 0;
            for (size_t i = 0; i < X::static_size(); i++) {
               value_type diff = (*x.get())[i] - (*y.get())[i];
               res += diff * diff;
            }
            z->value() = res / X::static_size();
         }
      }
   }

   // Compute dz/dx in grad
   template <class Gradient>
      requires(::ten::is_vector_v<Gradient> || ::ten::is_matrix_v<Gradient>)
   void gradient_left(X &x, Y &y, Gradient &grad) {
      if (x.size() != y.size()) {
         std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if (x.size() != grad.size()) {
            std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << x.size() << "\n";
         }
         using T = Gradient::value_type;
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = 2 * (x[i] - y[i]) / T(grad.size());
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_svector_v<Gradient> || ::ten::is_smatrix_v<Gradient>)
   void gradient_left(X &x, Y &y, Gradient &grad) {
      if constexpr (X::static_size() != Y::static_size()) {
         std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if constexpr (X::static_size() != Gradient::static_size()) {
            std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << X::static_size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < Gradient::static_size(); i++) {
               grad[i] = 2 * (x[i] - y[i]) / T(Gradient::static_size());
            }
         }
      }
   }

   // Compute dz/dy in grad
   template <class Gradient>
      requires(::ten::is_vector_v<Gradient> || ::ten::is_matrix_v<Gradient>)
   void gradient_right(X &x, Y &y, Gradient &grad) {
      if (x.size() != y.size()) {
         std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if (x.size() != grad.size()) {
            std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << x.size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < grad.size(); i++) {
               grad[i] = -2 * (x[i] - y[i]) / T(grad.size());
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_svector_v<Gradient> || ::ten::is_smatrix_v<Gradient>)
   void gradient_right(X &x, Y &y, Gradient &grad) {
      if (X::static_size() != Y::static_size()) {
         std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if constexpr (X::static_size() != Gradient::static_size()) {
            std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << X::static_size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < Gradient::static_size(); i++) {
               grad[i] = -2 * (x[i] - y[i]) / T(Gradient::static_size());
            }
         }
      }
   }
};

} // namespace ten::functional

#endif
