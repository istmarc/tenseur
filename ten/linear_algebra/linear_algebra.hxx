#ifndef TENSEUR_LINEAR_ALGEBRA
#define TENSEUR_LINEAR_ALGEBRA

#include <Ten/Types.hxx>
#include <Ten/Kernels/BlasAPI.hxx>

namespace ten{

/// Vector norms
enum class VectorNorm{
   L2 = 1,
   L1 = 2
};

/// Norm of a vector
template<class V>
requires(::ten::isVector<V>::value)
typename T::value_type norm(const V& v, const normType = VectorNorm::L2) {
}

/// Matrix norms
enum class MatrixNorm{
   Frobenius = 1,
};

/// Norm of a matrix
template<class M>
requires(::ten::isMatrix<M>::value)
typename M::value_type norm(const M& m, const normType = MatrixNorm::Frobenius) {
}

/// dot(a, b)
/// Dot porduct between two vectors
template<typename V>
requires(::ten::isVector<V>::value)
typename V::value_type dot(V& a, const V& b) {
   size_t n = a.size();
   return ten::kernels::blas::dot(n, a.data(), 1, b.data(), 1);
}

/// outer(a, b)
/// Outer product between two vectors
template<class V>
::ten::Matrix<typename V::value_type> outer(const V& a, const V& b) {
   using type = typename V::value_type;
   size_type n = a.size();
   size_type m = b.size();
   if (n != m) {
      std::cerr << "Tenseur: Outer product, different vector sizes.";
   }
   ::ten::Matrix<type> c = zeros({n, n});
   for (size_type i = 0; i < n; i++) {
      for (size_type j = 0; j < n; j++) {
         c(i, j) = a[i] * b[j];
      }
   }
   return c;
}

// TODO inv

// TODO pinv

// TODO determinant

// TODO Power

}

#endif
