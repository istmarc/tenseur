#ifndef TEN_KERNELS_LAPACK_API
#define TEN_KERNELS_LAPACK_API

#include <cstddef>

#include <ten/types.hxx>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif

namespace ten::kernels::lapack {

enum class transop : char { no = 'N', trans = 'T' };

static CBLAS_TRANSPOSE cast(const transop op) {
   if (op == transop::no) {
      return CBLAS_TRANSPOSE::CblasNoTrans;
   } else {
      return CBLAS_TRANSPOSE::CblasTrans;
   }
}

static int cast(const storage_order order) {
   if (order == storage_order::col_major) {
      return LAPACK_COL_MAJOR;
   } else {
      return LAPACK_ROW_MAJOR;
   }
}

/// QR factorization
template <typename T>
static void qr_fact(storage_order layout, size_t m, size_t n, T *a, size_t lda,
                    T *tau);

template <>
static void qr_fact(storage_order layout, size_t m, size_t n, float *a,
                    size_t lda, float *tau) {
   LAPACKE_sgeqrf(cast(layout), m, n, a, lda, tau);
};

template <typename T>
static void qr_factq(storage_order layout, size_t m, size_t n, size_t k, T *a,
                     size_t lda, T *tau);

template <>
static void qr_factq(storage_order layout, size_t m, size_t n, size_t k,
                     float *a, size_t lda, float *tau) {
   LAPACKE_sorgqr(cast(layout), m, n, k, a, lda, tau);
}

// LU factorization
template <typename T>
static void lu_fact(storage_order layout, size_t m, size_t n, T *a, size_t lda,
                    int *ipiv);

template <>
static void lu_fact(storage_order layout, size_t m, size_t n, float *a,
                    size_t lda, int *ipiv) {
   LAPACKE_sgetrf(cast(layout), m, n, a, lda, ipiv);
}

// Cholesky factorization
template <typename T>
static void cholesky_fact(storage_order layout, char uplo, size_t n, T *a,
                          size_t lda);

template <>
static void cholesky_fact(storage_order layout, char uplo, size_t n, float *a,
                          size_t lda) {
   LAPACKE_spotrf(cast(layout), uplo, n, a, lda);
}

// SVD decomposition
template <typename T>
static void svd_fact(storage_order layout, char jobu, char jobvt, size_t m,
                     size_t n, T *a, size_t lda, T *s, T *u, size_t ldu, T *vt,
                     size_t ldvt, T *work);

template <>
static void svd_fact(storage_order layout, char jobu, char jobvt, size_t m,
                     size_t n, float *a, size_t lda, float *s, float *u,
                     size_t ldu, float *vt, size_t ldvt, float *work) {
   LAPACKE_sgesvd(cast(layout), jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                  work);
}

// Inverse of a matrix
template<typename T>
static void inv(storage_order layout, size_t n, T* a, size_t lda, int32_t* ipiv);

template<>
static void inv(storage_order layout, size_t n, float* a, size_t lda, int32_t* ipiv) {
   LAPACKE_sgetrf(cast(layout), n, n, a, lda, ipiv);
   LAPACKE_sgetri(cast(layout), n, a, lda, ipiv);
}

} // namespace ten::kernels::lapack

#endif
