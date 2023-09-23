#ifndef TENSEUR_UTILS_HXX
#define TENSEUR_UTILS_HXX

#include <Ten/Types.hxx>

#include <array>
#include <fstream>
#include <sstream>
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

////////////////////////////////////////////////////////////////////////////////
// save to a file

template <class V> static void SaveVectorToFile(V &&vec, std::ofstream &file) {
   std::cout << "save vector to file" << std::endl;
   for (size_t i = 0; i < vec.size(); i++) {
      file << vec[i] << "\n";
   }
}

template <class M>
static void SaveMatrixToFile(M &&mat, std::ofstream &file,
                             const std::string &sep) {
   for (size_t i = 0; i < mat.dim(0); i++) {
      file << mat(i, 0);
      for (size_t j = 1; j < mat.dim(1); j++) {
         file << sep << mat(i, j);
      }
      file << "\n";
   }
}

template <typename T>
static void SaveToFile(T &&t, const std::string &fileName,
                       const FileType &filetype, std::string sep) {
   std::ofstream file;
   file.open(fileName);

   if (!file.is_open()) {
      std::stringstream ss;
      ss << "Tenseur : can't open file ";
      ss << fileName;
      throw std::runtime_error(ss.str());
   }

   using type = typename std::remove_cv_t<std::remove_reference_t<T>>;
   // if constexpr (std::is_same_v<typename type::value_type, float>){

   if constexpr (type::rank() == 1) {
      SaveVectorToFile(t, file);
   }
   if constexpr (type::rank() == 2) {
      SaveMatrixToFile(t, file, sep);
   }
   if constexpr (type::rank() > 2) {
      std::stringstream ss;
      ss << "Tenseur : SaveToFile(...) not implemented ";
      ss << "for tensors of size > 2";
      throw std::stringstream(ss.str());
   }
}

}; // namespace ten::details

#endif
