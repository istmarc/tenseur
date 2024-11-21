#ifndef TENSEUR_LINEAR_ALGEBRA_FACTORIZATION
#define TENSEUR_LINEAR_ALGEBRA_FACTORIZATION

#include <ten/kernels/lapack_api.hxx>
#include <ten/tensor.hxx>

namespace ten {

/// QR factorization
/// Factorize a matrix A = QR
template <class __t = float> class qr {
 private:
   matrix<__t> _q;
   matrix<__t> _r;

 public:
   qr(){};

   void factorize(const matrix<__t> &x) {
      // Copy x
      matrix<__t> a(x.shape());
      for (size_t i = 0; i < x.dim(0); i++) {
         for (size_t j = 0; j < x.dim(1); j++) {
            a(i, j) = x(i, j);
         }
      }

      size_t m = a.dim(0);
      size_t n = a.dim(1);
      auto layout = a.storage_order();
      _q = ::ten::zeros<matrix<__t>>({m, n});
      _r = ::ten::zeros<matrix<__t>>({n, n});

      ::ten::vector<__t> tau({n});
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
};

/// LU factorization
/// Factorize a matrix PA = LU
/// P i a permutation matrix, the inverse of P is its transpose
template <class __t = float> class lu {
 private:
   matrix<__t> _l;
   matrix<__t> _u;
   ::ten::vector<int32_t> _p;

 public:
   void factorize(const matrix<__t> &x) {
      // Copy x
      matrix<__t> a(x.shape());
      for (size_t i = 0; i < x.dim(0); i++) {
         for (size_t j = 0; j < x.dim(1); j++) {
            a(i, j) = x(i, j);
         }
      }

      size_t m = a.dim(0);
      size_t n = a.dim(1);

      if (m != n) {
         std::cout << "LU decomposition: input is not a square matrix"
                   << std::endl;
      }

      size_t q = std::min(m, n);
      _p = ::ten::vector<int32_t>({q});
      for (size_t i = 0; i < q; i++) {
         _p[i] = __t(i);
      }
      ::ten::vector<int32_t> ipiv({q});

      _l = ::ten::matrix<__t>({m, n});
      _u = ::ten::matrix<__t>({n, n});

      auto layout = a.storage_order();
      size_t lda = a.is_transposed() ? n : m;
      ::ten::kernels::lapack::lu_fact(layout, m, n, a.data(), lda, ipiv.data());

      // Copy L
      for (size_t i = 0; i < m; i++) {
         _l(i, i) = __t(1.);
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
      std::cout << ipiv << std::endl;
      std::cout << _p << std::endl;
   }

   auto l() const { return _l; }

   auto u() const { return _u; }

   auto p() const {
      matrix<__t> p = ::ten::zeros<matrix<__t>>({_l.dim(0), _l.dim(1)});
      std::cout << _p << std::endl;
      for (size_t i = 0; i < _p.size(); i++) {
         p(i, _p(i)) = 1.;
      }
      return p;
   }
};

// TODO SVD factorization
template <class __t = float> class svd {
 private:
   matrix<__t> _u;
   diagonal<__t> _sigma;
   matrix<__t> _vt;

 public:
   void factorize(matrix<__t> a) {}

   auto u() const { return _u; }

   auto sigma() const { return _sigma; }

   auto vt() const { return _vt; }
};

// TODO Cholesky
template <class __t = float> class cholesky {};

} // namespace ten

#endif
