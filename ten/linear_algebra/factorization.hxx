#ifndef TENSEUR_LINEAR_ALGEBRA_FACTORIZATION
#define TENSEUR_LINEAR_ALGEBRA_FACTORIZATION

#include <ten/kernels/lapack_api.hxx>
#include <ten/tensor.hxx>

namespace ten {

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

      matrix<__t> tau({n});
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

// TODO LU factorization
template <class __t = float> class lu {
 private:
   matrix<__t> __l;
   matrix<__t> __u;

 public:
   void factorize(matrix<__t> a) {}

   auto l() const { return __l; }

   auto u() const { return __u; }
};

// TODO SVD factorization
template <class __t = float> class svd {
 private:
   matrix<__t> __u;
   diagonal<__t> __sigma;
   matrix<__t> __vt;

 public:
   void factorize(matrix<__t> a) {}

   auto u() const { return __u; }

   auto sigma() const { return __sigma; }

   auto vt() const { return __vt; }
};

// Cholesky
template <class __t = float> class cholesky {};

} // namespace ten

#endif
