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

// asum
template<typename T>
static T asum(const int32_t n, const T* x, const int32_t incx);

template<>
float asum(const int32_t n, const float* x, const int32_t incx) {
   return cblas_sasum(n, x, incx);
}

template<>
double asum(const int32_t n, const double* x, const int32_t incx) {
   return cblas_dasum(n, x, incx);
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

template <>
void axpy(const int n, const double a, const double *x, const int incx, double *y,
          const int incy) {
   cblas_daxpy(n, a, x, incx, y, incy);
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

template <>
double dot(const int n, const double *x, const int incx, const double *y,
          const int incy) {
   return cblas_ddot(n, x, incx, y, incy);
}

// Vector matrix multiplication
// y = alpha * a * x + beta * y
template <typename T>
static void gemv(transop trans, const int m, const int n, const T alpha,
                 const T *a, const int lda, const T *x, const int incx,
                 const T beta, T *y, int incy);

template <>
void gemv(transop trans, const int m, const int n, const float alpha,
          const float *a, const int lda, const float *x, const int incx,
          const float beta, float *y, const int incy) {
   cblas_sgemv(CBLAS_ORDER::CblasColMajor, cast(trans), m, n, alpha, a, lda, x,
               incx, beta, y, incy);
}

template <>
void gemv(transop trans, const int m, const int n, const double alpha,
          const double *a, const int lda, const double *x, const int incx,
          const double beta, double *y, const int incy) {
   cblas_dgemv(CBLAS_ORDER::CblasColMajor, cast(trans), m, n, alpha, a, lda, x,
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

template <>
void gemm(transop transa, transop transb, const int m, const int n, const int k,
          const double alpha, const double *a, const int lda, const double *b,
          const int ldb, const double beta, double *c, const int ldc) {
   cblas_dgemm(CBLAS_ORDER::CblasColMajor, cast(transa), cast(transb), m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
}

} // namespace ten::kernels::blas

#endif
