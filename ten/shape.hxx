#ifndef TENSEUR_SHAPE_HXX
#define TENSEUR_SHAPE_HXX

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <ten/types.hxx>

namespace ten {

/// Dynamic dimension
static constexpr size_type dynamic = 0;

namespace details {
template <size_type __first, size_type... __rest> struct fold_mul {
   static constexpr size_type value = __first * fold_mul<__rest...>::value;
};
template <size_type __first> struct fold_mul<__first> {
   static constexpr size_type value = __first;
};

// Return true if at least one of the dimensions is dynamic
template <size_type __first, size_type... __tail> consteval bool is_dynamic() {
   if constexpr (__first == dynamic)
      return true;
   if constexpr (sizeof...(__tail) > 0)
      return is_dynamic<__tail...>();
   return false;
}

// Sequence of ::ten::dynamic
template <size_type... __index> struct dynamic_sequence {
   using type = dynamic_sequence<__index...>;
   using shape_type = ::ten::shape<__index...>;
};

// makeDynamicSequence<N> = DynamicSequence<dynamic, dynamic, ..., dynamic>
//                                                       N times
template <typename, typename> struct concat_dynamic_sequence;
template <size_type... __a, size_type... __b>
struct concat_dynamic_sequence<dynamic_sequence<__a...>,
                               dynamic_sequence<__b...>> {
   using type = dynamic_sequence<__a..., __b...>;
};

template <size_type> struct make_dynamic_sequence;
template <> struct make_dynamic_sequence<1> {
   using type = dynamic_sequence<dynamic>;
};
template <size_type __n> struct make_dynamic_sequence {
   using type = typename concat_dynamic_sequence<
       typename make_dynamic_sequence<__n - 1>::type,
       make_dynamic_sequence<1>::type>::type;
};

} // namespace details

/// \typedef dynamic_shape
/// Dynamic shape from rank
/// 1, Shape<dynamci>
/// 2, Shape<dynamic, dynamic>
/// ...
/// N, Shape<dynamic, ..., dynamic>
///                N times
template <size_type __rank>
using dynamic_shape =
    typename details::make_dynamic_sequence<__rank>::type::shape_type;

/// \class shape
///
/// Handles static, dynamic, and mixed shapes of a given rank.
/// Static dimensions refers to dimenions known at compile time. Dynamic
/// refers to dimensions known at runtime.
template <size_type __first, size_type... __rest> class shape {
 private:
   /// Rank (Number of dimensions)
   static constexpr size_type _rank = 1 + sizeof...(__rest);

   /// Chek if one of the dimensions is dynamic
   static constexpr bool _is_dynamic =
       details::is_dynamic<__first, __rest...>();

   /// Shape known at compile time
   static constexpr size_type _static_size{
       details::fold_mul<__first, __rest...>::value};

   /// Stores the dimensions known at compile time
   /// The shape is equal to 0 for dynamic shapes
   static constexpr std::array<size_type, _rank> _static_dims{__first,
                                                              __rest...};

   // Dynamic shape size
   size_type _size;
   /// Dynamic dimensions
   std::array<size_type, _rank> _dims{__first, __rest...};

 public:
   // For deserialization
   shape(size_type size, std::array<size_type, _rank> &&dims)
       : _size(size), _dims(std::move(dims)) {}

   shape() noexcept
      requires(!_is_dynamic)
   {}

   shape() noexcept {}

   shape(const shape& r) {
      _size = r._size;
      _dims = r._dims;
   }

   shape(shape&& r) {
      _size = std::move(r._size);
      _dims = std::move(r._dims);
   }

   shape& operator=(const shape& r) {
      _size = r._size;
      _dims = r._dims;
      return *this;
   }

   shape& operator=(shape&& r) {
      _size = std::move(r._size);
      _dims = std::move(r._dims);
      return *this;
   }

   /*explicit shape(std::vector<size_type> &&dims) noexcept
      requires(_is_dynamic)
   {
      size_type index = 0;
      _size = 1;
      for (auto value = dims.begin(); value != dims.end(); value++, index++) {
         _dims[index] = *value;
         _size *= *value;
      }
   }*/

   explicit shape(std::initializer_list<size_type> &&dims) noexcept
      requires(_is_dynamic)
   {
      size_type index = 0;
      _size = 1;
      for (auto value = dims.begin(); value != dims.end(); value++, index++) {
         _dims[index] = *value;
         _size *= *value;
      }
   }

   /// Returns the rank (number of dimensions)
   [[nodiscard]] inline static constexpr size_type rank() { return _rank; }

   /// Returns whether all the dimensions are statics
   [[nodiscard]] inline static constexpr bool is_static() {
      return !_is_dynamic;
   }

   /// Return whether all the dimensions are dynamics
   [[nodiscard]] inline static constexpr bool is_dynamic() {
      return _is_dynamic;
   }

   /// Returns the size known at compile time
   [[nodiscard]] inline static constexpr size_type static_size() {
      return _static_size;
   }

   /// Returns the size at runtime
   [[nodiscard]] inline size_type size() const {
      if constexpr (!_is_dynamic) {
         return _static_size;
      } else {
         return _size;
      }
   }

   /// Returns whether the dim is dynamic
   template <size_type __index>
   [[nodiscard]] inline static constexpr bool is_dynamic_dim() {
      return _static_dims[__index] == dynamic;
   }

   /// Returns whether the dim is static
   template <size_type __index>
   [[nodiscard]] inline static constexpr bool is_static_dim() {
      return _static_dims[__index] != dynamic;
   }

   /// Returns the static dimension at index
   template <size_type __index>
   [[nodiscard]] inline static constexpr size_type static_dim() {
      return _static_dims[__index];
   }

   /// Returns the dynamic dimension at index
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _dims[index];
   }

   /// Set the dynamic shape at index to value
   inline void set_dim(size_type index, size_type value)
      requires(_is_dynamic)
   {
      _dims[index] = value;
   }

   template <class __shape_type>
   // TODO requires __shape_type must be a shape type
   bool operator==(const __shape_type &s) const noexcept {
      if (__shape_type::rank() != rank()) {
         return false;
      }
      if (s.size() != size()) {
         return false;
      }
      for (size_t i = 0; i < __shape_type::rank(); i++) {
         if (s.dim(i) != dim(i)) {
            return false;
         }
      }
      return true;
   }

   // overload << operator
   template <size_type... Ts>
   friend std::ostream &operator<<(std::ostream &, const shape<Ts...> &);

   template <size_type... Ts>
   friend bool serialize(std::ostream &os, shape<Ts...> &s);

   template <class Shape>
      requires(::ten::is_shape<Shape>::value)
   friend Shape deserialize(std::istream &is);
};

template <size_type... Ts> bool serialize(std::ostream &os, shape<Ts...> &s) {
   os.write(reinterpret_cast<char *>(&s._size), sizeof(s._size));
   os.write(reinterpret_cast<char *>(s._dims.data()),
            s._rank * sizeof(s._dims[0]));

   return os.good();
}

template <class Shape>
   requires(::ten::is_shape<Shape>::value)
Shape deserialize(std::istream &is) {
   constexpr std::size_t rank = Shape::rank();
   std::size_t size;
   std::array<size_type, rank> dims;
   is.read(reinterpret_cast<char *>(&size), sizeof(size));
   is.read(reinterpret_cast<char *>(dims.data()), rank * sizeof(dims[0]));

   return Shape(size, std::move(dims));
}

namespace details {
template <size_type __n, size_type... __ts>
void print_shape(std::ostream &os, const shape<__ts...> &s) {
   using shape_type = shape<__ts...>;

   if constexpr (shape_type::template is_static_dim<__n>()) {
      os << shape_type::template static_dim<__n>();
   } else {
      os << s.dim(__n);
   }

   if constexpr (__n + 1 < shape_type::rank()) {
      os << "x";
      print_shape<__n + 1, __ts...>(os, s);
   }
}
} // namespace details

template <size_type... __ts>
std::ostream &operator<<(std::ostream &os, const shape<__ts...> &s) {
   details::print_shape<0, __ts...>(os, s);
   return os;
}

namespace details {
// Compute the Nth static stride
template <size_type __n, class __shape> struct nth_static_stride {
   static constexpr size_type value =
       nth_static_stride<__n - 1, __shape>::value *
       __shape::template static_dim<__n - 1>();
};
template <class __shape> struct nth_static_stride<0, __shape> {
   static constexpr size_type value = 1;
};

// Strides stored in an array
template <class __shape, class __index> struct static_stride_array;
template <class __shape, size_t... __index>
struct static_stride_array<__shape, std::index_sequence<__index...>> {
   static constexpr std::array<size_type, sizeof...(__index)> value{
       nth_static_stride<__index, __shape>::value...};
};

// Compute the static strides
template <class __shape, storage_order __order>
consteval auto compute_static_strides() {
   if constexpr (__order == storage_order::col_major) {
      return static_stride_array<
          __shape, std::make_index_sequence<__shape::rank()>>::value;
   }
   if constexpr (__order == storage_order::row_major) {
      std::array<size_type, __shape::rank()> res{};
      return res;
   }
}

// Dynamic strides
template <typename __shape, storage_order __order>
auto compute_strides(const __shape &dims) {
   constexpr size_type n = __shape::rank();
   std::array<size_type, n> strides{};
   if constexpr (__order == storage_order::row_major) {
      strides[n - 1] = 1;
      for (size_type i = n - 1; i > 0; i--) {
         strides[i - 1] = dims.dim(i) * strides[i];
      }
   }
   if constexpr (__order == storage_order::col_major) {
      strides[0] = 1;
      for (size_type i = 1; i < n; i++)
         strides[i] = dims.dim(i - 1) * strides[i - 1];
   }
   return strides;
}

} // namespace details

/// \class stride
/// Strides from shape
template <class __shape, storage_order __order> class stride {
 public:
   using shape_type = __shape;

 private:
   // Static strides
   static constexpr std::array<size_type, __shape::rank()> _static_strides{
       details::compute_static_strides<__shape, __order>()};

   // Dynamic strides
   std::array<size_type, __shape::rank()> _strides;

 public:
   stride(std::array<size_type, shape_type::rank()> &&dims)
       : _strides(std::move(dims)) {}

   stride() noexcept {}

   stride() noexcept
      requires(__shape::is_static())
   {}

   explicit stride(const __shape &shape) noexcept
      requires(__shape::is_dynamic())
       : _strides(details::compute_strides<__shape, __order>(shape)) {}

   template <size_type __index> static constexpr size_type static_dim() {
      return _static_strides[__index];
   }

   size_type dim(size_type index) const { return _strides[index]; }

   /// Returns the rank (number of dimensions)
   [[nodiscard]] inline static constexpr size_type rank() {
      return __shape::rank();
   }

   // Overload the << operator
   template <class __shape_type, storage_order __storage_type>
   friend std::ostream &
   operator<<(std::ostream &os, const stride<__shape_type, __storage_type> &);

   template <class ShapeType, storage_order StorageOrder>
   // requires(::ten::is_shape<ShapeType>::value)
   friend bool serialize(std::ostream &os, stride<ShapeType, StorageOrder> &s);

   template <class StrideType>
      requires(ten::is_stride<StrideType>::value)
   friend StrideType deserialize(std::istream &is);
};

template <class ShapeType, storage_order StorageOrder>
// requires(::ten::is_shape<ShapeType>::value)
bool serialize(std::ostream &os, stride<ShapeType, StorageOrder> &s) {
   constexpr size_t rank = stride<ShapeType, StorageOrder>::rank();
   os.write(reinterpret_cast<char *>(s._strides.data()),
            rank * sizeof(s._strides[0]));

   return os.good();
}

template <class StrideType>
   requires(::ten::is_stride<StrideType>::value)
StrideType deserialize(std::istream &is) {
   constexpr size_t rank = StrideType::rank();
   std::array<size_type, rank> dims;
   is.read(reinterpret_cast<char *>(dims.data()), rank * sizeof(dims[0]));

   return StrideType(std::move(dims));
}

namespace details {
template <size_type __n, class __shape, storage_order __order>
void print_stride(std::ostream &os, const stride<__shape, __order> &strides) {
   using stride_type = stride<__shape, __order>;

   if constexpr (__shape::is_static()) {
      os << stride_type::template static_dim<__n>();
   } else {
      os << strides.dim(__n);
   }
   if constexpr (__n + 1 < __shape::rank()) {
      os << "x";
      print_stride<__n + 1, __shape, __order>(os, strides);
   }
}
} // namespace details

template <class __shape, storage_order __order>
std::ostream &operator<<(std::ostream &os,
                         const stride<__shape, __order> &strides) {
   details::print_stride<0, __shape, __order>(os, strides);
   return os;
}

} // namespace ten
#endif
