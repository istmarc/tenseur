#ifndef TEN_KERNELS_BLAS_API_HXX
#define TEN_KERNELS_BLAS_API_HXX

#include <cstddef>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace ten::kernels::blas {

enum class transop : char { no = 'N', trans = 'T' };

static CBLAS_TRANSPOSE cast(const transop op) {
   if (op == transop::no) {
      return CBLAS_TRANSPOSE::CblasNoTrans;
   } else {
      return CBLAS_TRANSPOSE::CblasTrans;
   }
}

// Axpy
// y = a*x + y
template <typename T>
static void axpy(const int n, const T a, const T *x, const int incx, T *y,
                 const int incy);

template <>
void axpy(const int n, const float a, const float *x, const int incx, float *y,
          const int incy) {
   cblas_saxpy(n, a, x, incx, y, incy);
}

// Dot product of two vectors
// x * y
template <typename T>
static T dot(const int n, const T *x, const int incx, const T *y,
             const int incy);

template <>
float dot(const int n, const float *x, const int incx, const float *y,
          const int incy) {
   return cblas_sdot(n, x, incx, y, incy);
}

// Vector matrix multiplication
// y = alpha * a * x + beta * y
template <typename T>
static void gemv(transop trans, const int m, const int n, const T alpha,
                 const T *a, const int lda, const T *x, const int incx,
                 const T beta, float *y, int incy);

template <>
void gemv(transop trans, const int m, const int n, const float alpha,
          const float *a, const int lda, const float *x, const int incx,
          const float beta, float *y, const int incy) {
   cblas_sgemv(CBLAS_ORDER::CblasColMajor, cast(trans), m, n, alpha, a, lda, x,
               incx, beta, y, incy);
}

// General matrix multiplication
// c = alpha * a * b + beta * c
template <typename T>
static void gemm(transop transa, transop transb, const int m, const int n,
                 const int k, const T alpha, const T *a, const int lda,
                 const T *b, const int ldb, const T beta, T *c, const int ldc);

template <>
void gemm(transop transa, transop transb, const int m, const int n, const int k,
          const float alpha, const float *a, const int lda, const float *b,
          const int ldb, const float beta, float *c, const int ldc) {
   cblas_sgemm(CBLAS_ORDER::CblasColMajor, cast(transa), cast(transb), m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
}

} // namespace ten::kernels::blas

#endif
