#ifndef TENSEUR_LINEAR_ALGEBRA
#define TENSEUR_LINEAR_ALGEBRA

#include <ten/kernels/blas_api.hxx>
#include <ten/kernels/lapack_api.hxx>
#include <ten/linear_algebra/factorization.hxx>
#include <ten/types.hxx>

namespace ten::linalg {

/// Vector norms
enum class vector_norm { l2 = 1, l1 = 2, linf, lp };

/// Norm of a vector
template <class V>
   requires(::ten::is_vector<V>::value)
typename V::value_type norm(const V &v,
                            const vector_norm norm_type = vector_norm::l2) {
   using value_type = V::value_type;

   if (norm_type == vector_norm::l2) {
      value_type r = value_type(0);
      for (size_t i = 0; i < v.size(); i++) {
         r += v[i] * v[i];
      }
      return std::sqrt(r);
   }
   if (norm_type == vector_norm::l1) {
      value_type r = value_type(0);
      for (size_t i = 0; i < v.size(); i++) {
         r += std::abs(v[i]);
      }
      return r;
   }
   if (norm_type == vector_norm::linf) {
      value_type r = std::abs(v[0]);
      for (size_t i = 1; i < v.size(); i++) {
         r = std::max(r, std::abs(v[i]));
      }
      return r;
   }
}

template <class V>
   requires(::ten::is_vector<V>::value)
typename V::value_type pnorm(const V &v, const size_t p = 2) {
   using value_type = V::value_type;

   if (p == 0) {
      std::cerr << "Vector norm Lp: p is null" << std::endl;
   }
   value_type r = value_type(0);
   for (size_t i = 0; i < v.size(); i++) {
      r += std::pow(std::abs(v[i]), p);
   }
   r = std::pow(r, value_type(1) / value_type(p));
   return r;
}

/// Matrix norms
enum class matrix_norm {
   frobenius = 1,
   l1 = 2,
   linf = 3,
};

/// Norm of a matrix
template <class M>
   requires(::ten::is_matrix<M>::value)
typename M::value_type
norm(const M &m, const matrix_norm norm_type = matrix_norm::frobenius) {
   using value_type = M::value_type;

   if (norm_type == matrix_norm::frobenius) {
      value_type r = value_type(0);
      for (size_t i = 0; i < m.size(); i++) {
         r += std::abs(m[i]) * std::abs(m[i]);
      }
      return std::sqrt(r);
   }

   if (norm_type == matrix_norm::l1) {
      size_t p = m.dim(0);
      size_t q = m.dim(1);
      value_type r = std::abs(m(0, 0));
      for (size_t i = 1; i < p; i++) {
         r += std::abs(m(i, 0));
      }
      for (size_t j = 1; j < q; j++) {
         value_type s = std::abs(m(0, j));
         for (size_t i = 1; i < p; i++) {
            s += std::abs(m(i, j));
         }
         r = std::max(r, s);
      }
      return r;
   }

   if (norm_type == matrix_norm::linf) {
      size_t p = m.dim(0);
      size_t q = m.dim(1);
      value_type r = std::abs(m(0, 0));
      for (size_t j = 1; j < q; j++) {
         r += std::abs(m(0, j));
      }
      for (size_t i = 1; i < p; i++) {
         value_type s = std::abs(m(i, 0));
         for (size_t j = 1; j < q; j++) {
            s += std::abs(m(i, j));
         }
         r = std::max(r, s);
      }
      return r;
   }
}

/// dot(a, b)
/// Dot porduct between two vectors
template <typename V>
   requires(::ten::is_vector<V>::value)
typename V::value_type dot(V &a, V &b) {
   size_t n = a.size();
   return ten::kernels::blas::dot(n, a.data(), 1, b.data(), 1);
}

/// outer(a, b)
/// Outer product between two vectors
template <class V>
   requires(::ten::is_vector<V>::value)::ten::matrix<typename V::value_type>
outer(const V &a, const V &b) {
   using value_type = typename V::value_type;
   size_type n = a.size();
   size_type m = b.size();
   if (n != m) {
      std::cerr << "Outer product, different vector sizes.";
   }
   ::ten::matrix<value_type> c = zeros<ten::matrix<value_type>>({n, n});
   for (size_type i = 0; i < n; i++) {
      for (size_type j = 0; j < n; j++) {
         c(i, j) = a[i] * b[j];
      }
   }
   return c;
}

/// Compute the inverse of a matrix
template <class M>
   requires(ten::is_matrix<M>::value)
M inv(const M &a) {
   if (a.dim(0) != a.dim(1)) {
      std::cerr << "Matrix inverse: input is not square" << std::endl;
   }

   using value_type = M::value_type;
   matrix<value_type> x = a.copy();

   auto layout = x.storage_order();
   size_t m = x.dim(0);
   size_t n = x.dim(1);
   size_t lda = x.is_transposed() ? n : m;
   ::ten::vector<int32_t> ipiv({n});

   ::ten::kernels::lapack::inv(layout, n, x.data(), lda, ipiv.data());

   return x;
}

// Determinant
template <class M>
   requires(ten::is_matrix<M>::value)
typename M::value_type det(const M &m) {
   using value_type = M::value_type;

   ten::linalg::lu<value_type> lufact;
   lufact.factorize(m);
   auto [p, l, u] = lufact.factors();

   value_type d = 1.;
   // Number of rows exchange
   size_t n = 0;
   // Find the determinant of p
   for (size_t i = 0; i < m.dim(0); i++) {
      if (p(i, i) == 0.) {
         n += 1;
      }
   }
   if (n % 2 == 1) {
      d = -1;
   }
   std::cout << "det P  = " << d << std::endl;
   // Multiply by det(l) and det(u)
   for (size_t i = 0; i < m.dim(0); i++) {
      d *= l(i, i) * u(i, i);
   }
   return d;
}

// TODO pinv

// TODO Power

} // namespace ten::linalg

#endif
