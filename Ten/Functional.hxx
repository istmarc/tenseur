#ifndef TENSEUR_FUNCTIONAL_HXX
#define TENSEUR_FUNCTIONAL_HXX

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <type_traits>

#include <Ten/Kernels/Host>
#include <Ten/Types.hxx>

namespace ten {
enum class BinaryOperation;
}

namespace ten::functional {
// Functions types
template <bool params = false> struct Func {};

template <class T> struct HasParams {
   static constexpr bool value = std::is_base_of_v<Func<true>, T>;
};

////////////////////////////////////////////////////////////////////////////////
// Unary functions

/// Square root
template <class A, class B = A> struct Sqrt : Func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::sqrt(static_cast<value_type>(a[i]));
      }
   }
};

/// Absolute value
template <class A, class B = A> struct Abs : Func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::abs(static_cast<value_type>(a[i]));
      }
   }
};

/// Power
template <class A, class B = A> struct Pow : Func<true> {
 private:
   double _n;

 public:
   using output_type = B;

   explicit Pow(double n) : _n(n) {}

   void operator()(const A &a, B &b) const {
      using value_type = typename B::value_type;
      for (size_t i = 0; i < a.size(); i++) {
         b[i] = std::pow(static_cast<value_type>(a[i]), _n);
      }
   }
};

template <class A, class B = typename A::scalarnode_type> struct Min : Func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename A::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::min(static_cast<type>(a[i]), res);
      }
      b = res;
   }
};

template <class A, class B = typename A::scalarnode_type> struct Max : Func<> {
   using output_type = B;

   void operator()(const A &a, B &b) {
      using type = typename A::value_type;
      type res = a[0];
      for (size_t i = 1; i < a.size(); i++) {
         res = std::max(static_cast<type>(a[i]), res);
      }
      b = res;
   }
};

////////////////////////////////////////////////////////////////////////////////
// Binary functions (Add, Sub, Mul and Div)
namespace details {
template <class, class> struct CommonType;

template <::ten::traits::TensorNode A, ::ten::traits::TensorNode B>
   requires SameShape<A, B> && SameStorageOrder<A, B> && SameStorage<A, B> &&
            SameAllocator<A, B>
struct CommonType<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       TensorNode<value_type, typename A::shape_type, A::storageOrder(),
                  typename A::storage_type, typename A::allocator_type>;
};

template <class A, class B>
using common_type_t = typename CommonType<A, B>::type;
} // namespace details

// Binary function
template <::ten::BinaryOperation kind> struct BinaryFunc {

   template <class A, class B, class C = details::common_type_t<A, B>>
   struct Func : ::ten::functional::Func<> {

      using output_type = C;

      static constexpr typename C::shape_type
      outputShape(const typename A::shape_type &left,
                  const typename B::shape_type &right) {
         typename C::shape_type s(left);
         return s;
      }

      static auto outputShape(const A &a, const B &b) { return a.shape(); }

      void operator()(const A &left, const B &right, C &result) {
         ::ten::kernels::binaryOps<kind>(left, right, result);
      }
   };
};

namespace details {
template <class, class> struct MulResult;

// vector * vector
template <VectorNode A, VectorNode B>
   requires SameShape<A, B> && SameStorageOrder<A, B> && SameStorage<A, B> &&
            SameAllocator<A, B>
struct MulResult<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type =
       TensorNode<value_type, typename A::shape_type, A::storageOrder(),
                  typename A::storage_type, typename A::allocator_type>;
};

// matrix * matrix
template <MatrixNode A, MatrixNode B>
   requires SameStorageOrder<A, B> && SameStorage<A, B> && SameAllocator<A, B>
struct MulResult<A, B> {
   using value_type =
       std::common_type_t<typename A::value_type, typename B::value_type>;
   using type = TensorNode<value_type,
                           Shape<A::shape_type::template staticDim<0>(),
                                 B::shape_type::template staticDim<1>()>,
                           A::storageOrder(), typename A::storage_type,
                           typename A::allocator_type>;
};

// scalar * tensor
template <traits::ScalarNode A, traits::TensorNode B> struct MulResult<A, B> {
   using type = B;
};

} // namespace details

template <class A, class B, class C = typename details::MulResult<A, B>::type>
struct Mul;

// vector * vector
template <VectorNode X, VectorNode Y, VectorNode Z> struct Mul<X, Y, Z> {

   template <VectorNode A, VectorNode B,
             VectorNode C = typename details::MulResult<A, B>::type>
   using Func = BinaryFunc<::ten::BinaryOperation::mul>::Func<A, B, C>;
};

// matrix * matrix
template <MatrixNode X, MatrixNode Y, MatrixNode Z> struct Mul<X, Y, Z> {

   template <MatrixNode A, MatrixNode B,
             MatrixNode C = typename details::MulResult<A, B>::type>
   struct Func : ::ten::functional::Func<> {
      using output_type = C;

      static typename C::shape_type
      outputShape(const typename A::shape_type &left,
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

// scalar * tensor
template <traits::ScalarNode X, traits::TensorNode Y, traits::TensorNode Z>
struct Mul<X, Y, Z> {

   template <traits::ScalarNode A, traits::TensorNode B,
             traits::TensorNode C = typename details::MulResult<A, B>::type>
   struct Func : ::ten::functional::Func<> {
      using output_type = C;

      static typename C::shape_type
      outputShape(const typename B::shape_type &right) {
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

namespace details {
// dynamic reshape result
template <class A, class Shape> struct ReshapeResult {
   using type =
       TensorNode<typename A::value_type, Shape, A::storageOrder(),
                  typename A::storage_type, typename A::allocator_type>;
};
} // namespace details

// Reshape to static
template <class S> struct StaticReshape {
   static_assert(S::isStatic(), "Shape must be static");

   template <class A, class B = typename details::ReshapeResult<A, S>::type>
   struct Func : ::ten::functional::Func<> {
      using output_type = B;

      void operator()(const A &left, B &right) { right = B(left.storage()); }
   };
};

template <size_type... dims>
using DimsStaticReshape = StaticReshape<::ten::Shape<dims...>>;

// Dynamic reshape
template <class Shape> struct DynamicReshape {

   template <class A, class B = typename details::ReshapeResult<A, Shape>::type>
   struct Func : ::ten::functional::Func<true> {
    public:
      using output_type = B;

    private:
      Shape _shape;

    public:
      Func(const Shape &shape) : _shape(shape) {}
      Func(Shape &&shape) : _shape(std::move(shape)) {}

      typename B::shape_type outputShape(const typename A::shape_type &) {
         return _shape;
      }

      void operator()(const A &left, B &right) {
         right = B(left.storage(), _shape);
      }
   };
};

} // namespace ten::functional

#endif
