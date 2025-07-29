#ifndef TENSEUR_LINEAR_ALGEBRA_FACTORIZATION
#define TENSEUR_LINEAR_ALGEBRA_FACTORIZATION

#include <ten/kernels/lapack_api.hxx>
#include <ten/tensor.hxx>

namespace ten {
namespace linalg {

/// QR factorization
/// Factorize a matrix A = QR
template <class T = float> class qr_fact {
 public:
   using value_type = T;

 private:
   matrix<value_type> _q;
   matrix<value_type> _r;

 public:
   qr_fact() {};

   void factorize(const matrix<value_type> &x) {
      // Copy x
      matrix<value_type> a = x.copy();

      size_t m = a.dim(0);
      size_t n = a.dim(1);
      auto layout = a.storage_order();
      _q = ::ten::zeros<matrix<value_type>>({m, n});
      _r = ::ten::zeros<matrix<value_type>>({n, n});

      ::ten::vector<value_type> tau({n});
      size_t lda = a.is_transposed() ? n : m;
      ::ten::kernels::lapack::qr_fact(layout, m, n, a.data(), lda, tau.data());
      // Copy R
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i; j < n; j++) {
            _r(i, j) = a(i, j);
         }
      }
      // Copy Q
      size_t k = n;
      ::ten::kernels::lapack::qr_factq(layout, m, n, k, a.data(), lda,
                                       tau.data());
      for (size_t i = 0; i < n; i++) {
         for (size_t j = 0; j < n; j++) {
            _q(i, j) = a(i, j);
         }
      }
   }

   auto q() const { return _q; }

   auto r() const { return _r; }

   auto factors() const { return std::make_tuple(_q, _r); }
};

/// QR facttorization of a matrix, returns the factors
template<Matrix T>
auto qr(T&& A) -> decltype(auto) {
   using value_type = typename std::remove_cvref_t<T>::value_type;
   qr_fact<value_type> qr;
   qr.factorize(A);
   return qr.factors();
}

/// LU factorization
/// Factorize a matrix PA = LU
/// P i a permutation matrix, the inverse of P is its transpose
template <class T = float> class lu_fact {
 public:
   using value_type = T;

 private:
   matrix<value_type> _l;
   matrix<value_type> _u;
   ::ten::vector<int32_t> _p;

 public:
   lu_fact() {}

   void factorize(const matrix<value_type> &x) {
      // Copy x
      matrix<value_type> a = x.copy();

      size_t m = a.dim(0);
      size_t n = a.dim(1);

      if (m != n) {
         std::cerr << "LU decomposition: input is not a square matrix"
                   << std::endl;
      }

      size_t q = std::min(m, n);
      _p = ::ten::vector<int32_t>({q});
      for (size_t i = 0; i < q; i++) {
         _p[i] = value_type(i);
      }
      ::ten::vector<int32_t> ipiv({q});

      _l = ::ten::matrix<value_type>({m, n});
      _u = ::ten::matrix<value_type>({n, n});

      auto layout = a.storage_order();
      size_t lda = a.is_transposed() ? n : m;
      ::ten::kernels::lapack::lu_fact(layout, m, n, a.data(), lda, ipiv.data());

      // Copy L
      for (size_t i = 0; i < m; i++) {
         _l(i, i) = value_type(1.);
      }
      for (size_t i = 1; i < m; i++) {
         for (size_t j = 0; j < i; j++) {
            _l(i, j) = a(i, j);
         }
      }

      // Copy U
      for (size_t i = 0; i < m; i++) {
         for (size_t j = i; j < n; j++) {
            _u(i, j) = a(i, j);
         }
      }

      // Set _p
      for (size_t i = 0; i < q; i++) {
         std::swap(_p[i], _p[ipiv[i] - 1]);
      }
   }

   auto l() const { return _l; }

   auto u() const { return _u; }

   auto p() const {
      matrix<value_type> p =
          ::ten::zeros<matrix<value_type>>({_l.dim(0), _l.dim(1)});
      for (size_t i = 0; i < _p.size(); i++) {
         p(_p(i), i) = 1.;
      }
      return p;
   }

   auto factors() const { return std::make_tuple(p(), _l, _u); }
};

/// LU factorization of a matrix, returns the factors
template<Matrix T>
auto lu(T&& A) -> decltype(auto) {
   using value_type = typename std::remove_cvref_t<T>::value_type;
   lu_fact<value_type> lu;
   lu.factorize(A);
   return lu.factors();
}

// Cholesky factorization
// Factorize a matrix A = LU = L L^T = U^T U
template <class T = float> class cholesky_fact {
 public:
   using value_type = T;

 private:
   matrix<value_type> _l;
   matrix<value_type> _u;

 public:
   cholesky_fact() {}

   void factorize(const matrix<value_type> &a) {
      matrix<value_type> x = a.copy();

      size_t m = x.dim(0);
      size_t n = x.dim(1);
      if (m != n) {
         std::cerr << "Cholesky: input matrix is not square" << std::endl;
      }

      auto layout = x.storage_order();
      size_t lda = x.is_transposed() ? n : m;
      ::ten::kernels::lapack::cholesky_fact(layout, 'L', n, x.data(), lda);

      _l = ::ten::zeros<ten::matrix<value_type>>({n, n});
      _u = ::ten::zeros<ten::matrix<value_type>>({n, n});

      // Copy L
      for (size_t i = 0; i < m; i++) {
         for (size_t j = 0; j < i + 1; j++) {
            _l(i, j) = x(i, j);
         }
      }
      // Copy U
      for (size_t i = 0; i < m; i++) {
         for (size_t j = 0; j < i + 1; j++) {
            _u(j, i) = x(i, j);
         }
      }
   }

   auto l() const { return _l; }

   auto u() const { return _u; }

   auto factors() const { return std::make_tuple(_l, _u); }
};


/// Cholesky factorization of a matrix, returns the factors
template<Matrix T>
auto cholesky(T&& A) -> decltype(auto) {
   using value_type = typename std::remove_cvref_t<T>::value_type;
   cholesky_fact<value_type> chol;
   chol.factorize(A);
   return chol.factors();
}

/// SVD factorization
/// Factorize a matrix A = U * Sigma * V^T
template <class T = float> class svd_fact {
 public:
   using value_type = T;

 private:
   matrix<value_type> _u;
   diagonal<value_type> _sigma;
   matrix<value_type> _vt;

 public:
   svd_fact() {}

   void factorize(const matrix<value_type> &a) {
      matrix<value_type> x = a.copy();

      size_t m = x.dim(0);
      size_t n = x.dim(1);
      auto layout = x.storage_order();
      size_t lda = x.is_transposed() ? n : m;

      _u = zeros<ten::matrix<value_type>>({m, m});
      _sigma = ::ten::diagonal<value_type>({m, n});
      _vt = zeros<ten::matrix<value_type>>({n, n});

      ::ten::vector<value_type> work({m * n});

      ::ten::kernels::lapack::svd_fact(layout, 'A', 'A', m, n, x.data(), lda,
                                       _sigma.data(), _u.data(), m, _vt.data(),
                                       n, work.data());
   }

   auto u() const { return _u; }

   auto sigma() const { return _sigma; }

   auto vt() const { return _vt; }

   auto factors() const { return std::make_tuple(_u, _sigma, _vt); }
};

/// SVD factorization of a matrix, returns the factors
template<Matrix T>
auto svd(T&& A) -> decltype(auto) {
   using value_type = typename std::remove_cvref_t<T>::value_type;
   svd_fact<value_type> svd;
   svd.factorize(A);
   return svd.factors();
}

} // namespace linalg
} // namespace ten

#endif
