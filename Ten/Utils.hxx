#ifndef TENSEUR_UTILS_HXX
#define TENSEUR_UTILS_HXX

#include <Ten/Types.hxx>

#include <array>
#include <type_traits>

namespace ten {
template <class> std::string to_string();

template <> std::string to_string<float>() { return "float"; }

template <> std::string to_string<double>() { return "double"; }
} // namespace ten

namespace ten::details {

/*
// Generate indices in lexicographic order
template <typename _d>
[[nodiscard]] consteval auto generate_indices() -> decltype(auto) {
  using std::array;

  constexpr size_t order = _d::order();
  // sizeof...(_index);
  constexpr size_t size = _d::size();
  //(_index * ...);
  if constexpr (order == 1) {
    array<size_t, size> indices{};
    for (size_t i = 0; i < size; i++) {
      indices[i] = i;
    }
    return indices;
  } else {
    constexpr array<size_t, order> shape{_d::shape()};
    // {_index...};
    constexpr array<size_t, order> strides{compute_strides<_d>()};
    array<array<size_t, order>, size> indices{};
    for (size_t dim = 0; dim < order; dim++) {
      for (size_t k = 0; k < size; k += shape[dim] * strides[dim]) {
        for (size_t i = 0; i < shape[dim]; i++) {
          for (size_t j = k + i * strides[dim]; j < k + (i + 1) * strides[dim];
               j++) {
            indices[j][dim] = i;
          }
        }
      }
    }
    return indices;
  }
}*/

/// \fn linearIndex
/// Compute the linear index from the stride and the indices
template <class Stride>
[[nodiscard]] inline ::ten::size_type
linearIndex(const Stride &stride,
            const std::array<::ten::size_type, Stride::rank()> &indices) {
   ::ten::size_type index = 0;
   for (::ten::size_type i = 0; i < Stride::rank(); i++)
      index += indices[i] * stride.dim(i);
   return index;
}

/// \fn staticLinearIndex
/// Compute the linear index from the static strides and the indices
template <class Stride, size_type N = 0>
[[nodiscard]] size_type
staticLinearIndex(const std::array<::ten::size_type, Stride::rank()> &indices,
                  size_type index = 0) {
   index += indices[N] * Stride::template staticDim<N>();
   if constexpr (N + 1 < Stride::rank()) {
      return staticLinearIndex<Stride, N + 1>(indices, index);
   } else {
      return index;
   }
}

}; // namespace ten::details

#endif
