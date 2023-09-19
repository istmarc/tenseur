#ifndef TENSEUR_SHAPE_HXX
#define TENSEUR_SHAPE_HXX

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <Ten/Types.hxx>

namespace ten {

/// Dynamic dimension
static constexpr size_type dynamic = 0;

namespace details {
template <size_type First, size_type... Rest> struct FoldMul {
   static constexpr size_type value = First * FoldMul<Rest...>::value;
};
template <size_type First> struct FoldMul<First> {
   static constexpr size_type value = First;
};

// Return true if at least one of the dimensions is dynamic
template <size_type First, size_type... Tail> consteval bool isDynamic() {
   if constexpr (First == dynamic)
      return true;
   if constexpr (sizeof...(Tail) > 0)
      return isDynamic<Tail...>();
   return false;
}

// Sequence of ::ten::dynamic
template <size_type... I> struct DynamicSequence {
   using type = DynamicSequence<I...>;
   using shape_type = ::ten::Shape<I...>;
};

// makeDynamicSequence<N> = DynamicSequence<dynamic, dynamic, ..., dynamic>
//                                                       N times
template <typename, typename> struct ConcatDynamicSequence;
template <size_type... A, size_type... B>
struct ConcatDynamicSequence<DynamicSequence<A...>, DynamicSequence<B...>> {
   using type = DynamicSequence<A..., B...>;
};

template <size_type> struct MakeDynamicSequence;
template <> struct MakeDynamicSequence<1> {
   using type = DynamicSequence<dynamic>;
};
template <size_type N> struct MakeDynamicSequence {
   using type =
       typename ConcatDynamicSequence<typename MakeDynamicSequence<N - 1>::type,
                                      MakeDynamicSequence<1>::type>::type;
};

} // namespace details

/// \typedef DynamicShape
/// Dynamic shape from rank
/// 1, Shape<dynamci>
/// 2, Shape<dynamic, dynamic>
/// ...
/// N, Shape<dynamic, ..., dynamic>
///                N times
template <size_type Rank>
using DynamicShape =
    typename details::MakeDynamicSequence<Rank>::type::shape_type;

/// \class Shape
///
/// Handles static, dynamic, and mixed shapes of a given rank.
/// Static dimensions refers to dimenions known at compile time. Dynamic
/// refers to dimensions known at runtime.
template <size_type First, size_type... Rest> class Shape {
 private:
   /// Rank (Number of dimensions)
   static constexpr size_type _rank = 1 + sizeof...(Rest);

   /// Chek if one of the dimensions is dynamic
   static constexpr bool _isDynamic = details::isDynamic<First, Rest...>();

   /// Shape known at compile time
   static constexpr size_type _staticSize{
       details::FoldMul<First, Rest...>::value};

   /// Stores the dimensions known at compile time
   /// The shape is equal to 0 for dynamic shapes
   // static constexpr auto _staticDims = std::make_tuple(First, Rest...);
   static constexpr std::array<size_type, _rank> _staticDims{First, Rest...};

   // Dynamic shape size
   size_type _size;
   /// Dynamic dimensions
   std::array<size_type, _rank> _dims{First, Rest...};

 public:
   Shape() noexcept
      requires(!_isDynamic)
   {}

   explicit Shape(std::initializer_list<size_type> &&dims) noexcept
      requires(_isDynamic)
   {
      size_type index = 0;
      _size = 1;
      for (auto value = dims.begin(); value != dims.end(); value++, index++) {
         _dims[index] = *value;
         _size *= *value;
      }
   }

   // TODO Construct of Ten::Shape from an array of dimensions
   /*explicit Shape(const std::array<size_type, fRank>& dims) noexcept :
     fDynamicDims(dims) {
      fDynamicSize = 1;
      for (size_type index = 0; index < fRank; index++)
         fDynamicSize *= dims[index];
   }
   // TODO Construct of Ten::Shape from an array
   explicit Shape(std::array<size_type, fRank>&& dims) noexcept {
      fDynamicSize = 1;
      for (size_type index = 0; index < fRank; index++)
         fDynamicSize *= dims[index];
      fDynamicDims = std::move(dims);
   }*/

   /// Returns the rank (number of dimensions)
   [[nodiscard]] inline static constexpr size_type rank() { return _rank; }

   /// Returns whether all the dimensions are statics
   [[nodiscard]] inline static constexpr bool isStatic() { return !_isDynamic; }

   /// Return whether all the dimensions are dynamics
   [[nodiscard]] inline static constexpr bool isDynamic() { return _isDynamic; }

   /// Returns the size known at compile time
   [[nodiscard]] inline static constexpr size_type staticSize() {
      return _staticSize;
   }

   /// Returns the size at runtime
   [[nodiscard]] inline size_type size() const {
      if constexpr (!_isDynamic) {
         return _staticSize;
      } else {
         return _size;
      }
   }

   /// Returns whether the dim is dynamic
   template <size_type index>
   [[nodiscard]] inline static constexpr bool isDynamicDim() {
      // return std::get<index>(_staticDims) == dynamic;
      return _staticDims[index] == dynamic;
   }

   /// Returns whether the dim is static
   template <size_type index>
   [[nodiscard]] inline static constexpr bool isStaticDim() {
      return _staticDims[index] != dynamic;
   }

   /// Returns the static dimension at index
   template <size_type index>
   [[nodiscard]] inline static constexpr size_type staticDim() {
      // return std::get<index>(_staticDims);
      return _staticDims[index];
   }

   /// Returns the dynamic dimension at index
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _dims[index];
   }

   /// Set the dynamic shape at index to value
   inline void setDim(size_type index, size_type value)
      requires(_isDynamic)
   {
      _dims[index] = value;
   }

   template <class S>
   // TODO S must be a shape type
   bool operator==(const S &s) const noexcept {
      if (S::rank() != rank()) {
         return false;
      }
      if (s.size() != size()) {
         return false;
      }
      for (size_t i = 0; i < S::rank(); i++) {
         if (s.dim(i) != dim(i)) {
            return false;
         }
      }
      return true;
   }

   // overload << operator
   template <size_type... Ts>
   friend std::ostream &operator<<(std::ostream &, const Shape<Ts...> &);
};

namespace details {
template <size_type N, size_type... Ts>
void printShape(std::ostream &os, const Shape<Ts...> &shape) {
   using shape_type = Shape<Ts...>;

   if constexpr (shape_type::template isStaticDim<N>()) {
      os << shape_type::template staticDim<N>();
   } else {
      os << shape.dim(N);
   }

   if constexpr (N + 1 < shape_type::rank()) {
      os << "x";
      printShape<N + 1, Ts...>(os, shape);
   }
}
} // namespace details

template <size_type... Ts>
std::ostream &operator<<(std::ostream &os, const Shape<Ts...> &shape) {
   details::printShape<0, Ts...>(os, shape);
   return os;
}

namespace details {
// Compute the Nth static stride
template <size_type N, class S> struct NthStaticStride {
   static constexpr size_type value =
       NthStaticStride<N - 1, S>::value * S::template staticDim<N - 1>();
};
template <class S> struct NthStaticStride<0, S> {
   static constexpr size_type value = 1;
};

// Strides stored in an array
template <class S, class I> struct StaticStrideArray;
template <class S, size_t... I>
struct StaticStrideArray<S, std::index_sequence<I...>> {
   static constexpr std::array<size_type, sizeof...(I)> value{
       NthStaticStride<I, S>::value...};
};

// Compute the static strides
template <class S, StorageOrder order> consteval auto computeStaticStrides() {
   if constexpr (order == StorageOrder::ColMajor) {
      return StaticStrideArray<S, std::make_index_sequence<S::rank()>>::value;
   }
   if constexpr (order == StorageOrder::RowMajor) {
      std::array<size_type, S::rank()> res{};
      return res;
   }
}

// Dynamic strides
template <typename S, StorageOrder order> auto computeStrides(const S &shape) {
   constexpr size_type n = S::rank();
   std::array<size_type, n> strides{};
   if constexpr (order == StorageOrder::RowMajor) {
      strides[n - 1] = 1;
      for (size_type i = n - 1; i > 0; i--) {
         strides[i - 1] = shape.dim(i) * strides[i];
      }
   }
   if constexpr (order == StorageOrder::ColMajor) {
      strides[0] = 1;
      for (size_type i = 1; i < n; i++)
         strides[i] = shape.dim(i - 1) * strides[i - 1];
   }
   return strides;
}

} // namespace details

/// \class Strides
/// Strides from shape
template <class S, StorageOrder order> class Stride {
 public:
   using shape_type = S;

 private:
   // Static strides
   static constexpr std::array<size_type, S::rank()> _staticStrides{
       details::computeStaticStrides<S, order>()};

   // Dynamic strides
   std::array<size_type, S::rank()> _strides;

 public:
   Stride() noexcept
      requires(S::isStatic())
   {}

   explicit Stride(const S &shape) noexcept
      requires(S::isDynamic())
       : _strides(details::computeStrides<S, order>(shape)) {}

   template <size_type Index> static constexpr size_type staticDim() {
      return _staticStrides[Index];
   }

   size_type dim(size_type index) const { return _strides[index]; }

   /// Returns the rank (number of dimensions)
   [[nodiscard]] inline static constexpr size_type rank() { return S::rank(); }

   // Overload the << operator
   template <class T, StorageOrder R>
   friend std::ostream &operator<<(std::ostream &os, const Stride<S, R> &);
};

namespace details {
template <size_type N, class S, StorageOrder order>
void printStride(std::ostream &os, const Stride<S, order> &strides) {
   using stride_type = Stride<S, order>;

   if constexpr (S::isStatic()) {
      os << stride_type::template staticDim<N>();
   } else {
      os << strides.dim(N);
   }
   if constexpr (N + 1 < S::rank()) {
      os << "x";
      printStride<N + 1, S, order>(os, strides);
   }
}
} // namespace details

template <class S, StorageOrder order>
std::ostream &operator<<(std::ostream &os, const Stride<S, order> &strides) {
   details::printStride<0, S, order>(os, strides);
   return os;
}

} // namespace ten
#endif
