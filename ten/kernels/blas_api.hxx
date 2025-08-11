#ifndef TEN_KERNELS_BLAS_API_HXX
#define TEN_KERNELS_BLAS_API_HXX

#include <cstddef>
#include <complex>

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

// Copy
template<typename T>
static void copy(const int32_t n, const T* x, const int32_t incx, T* y, const int32_t incy);

template<>
void copy(const int32_t n, const float* x, const int32_t incx, float* y, const int32_t incy) {
   cblas_scopy(n, x, incx, y, incy);
}

template<>
void copy(const int32_t n, const double* x, const int32_t incx, double* y, const int32_t incy) {
   cblas_dcopy(n, x, incx, y, incy);
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

// Conjugate dot product
template<typename T>
static std::complex<T> dotc(const int32_t n, const std::complex<T>* x, const int32_t incx,
      const std::complex<T>* y, const int32_t incy);

template<>
std::complex<float> dotc(const int32_t n, const std::complex<float>* x, const int32_t incx,
   const std::complex<float>* y, const int32_t incy) {
   std::complex<float> c;
   cblas_cdotc_sub(n, x, incx, y, incy, &c);
   return c;
}

template<>
std::complex<double> dotc(const int32_t n, const std::complex<double>* x, const int32_t incx,
   const std::complex<double>* y, const int32_t incy) {
   std::complex<double> c;
   cblas_cdotc_sub(n, x, incx, y, incy, &c);
   return c;
}

// iamax
template<typename T>
static size_t iamax(const int32_t n, const T* x, const int32_t incx);

template<>
size_t iamax(const int32_t n, const float* x, const int32_t incx) {
   return cblas_isamax(n, x, incx);
}

template<>
size_t iamax(const int32_t n, const double* x, const int32_t incx) {
   return cblas_idamax(n, x, incx);
}

// nrm2
template<typename T>
static T nrm2(const int32_t n, const T* x, const int32_t incx);

template<>
float nrm2(const int32_t n, const float* x, const int32_t incx) {
   return cblas_snrm2(n, x, incx);
}

template<>
double nrm2(const int32_t n, const double* x, const int32_t incx) {
   return cblas_dnrm2(n, x, incx);
}

// scal
template<typename ScalarType, typename T>
static void scal(const int32_t n, const ScalarType alpha, T* x, const int32_t incx);

template<>
void scal(const int32_t n, const float alpha, float* x, const int32_t incx) {
   cblas_sscal(n, alpha, x, incx);
}

template<>
void scal(const int32_t n, const double alpha, double* x, const int32_t incx) {
   cblas_dscal(n, alpha, x, incx);
}

template<>
void scal(const int32_t n, const std::complex<float> alpha, std::complex<float>* x, const int32_t incx) {
   cblas_cscal(n, &alpha, x, incx);
}

template<>
void scal(const int32_t n, const std::complex<double> alpha, std::complex<double>* x, const int32_t incx) {
   cblas_zscal(n, &alpha, x, incx);
}

template<>
void scal(const int32_t n, const float alpha, std::complex<float>* x, const int32_t incx) {
   cblas_csscal(n, alpha, x, incx);
}

template<>
void scal(const int32_t n, const double alpha, std::complex<double>* x, const int32_t incx) {
   cblas_zdscal(n, alpha, x, incx);
}

// swap
template<typename T>
void swap(const int32_t n, T* x, const int32_t incx, T* y, const int32_t incy);

template<>
void swap(const int32_t n, float* x, const int32_t incx, float* y, const int32_t incy) {
   cblas_sswap(n, x, incx, y, incy);
}

template<>
void swap(const int32_t n, double* x, const int32_t incx, double* y, const int32_t incy) {
   cblas_dswap(n, x, incx, y, incy);
}

template<>
void swap(const int32_t n, std::complex<float>* x, const int32_t incx, std::complex<float>* y, const int32_t incy) {
   cblas_cswap(n, x, incx, y, incy);
}

template<>
void swap(const int32_t n, std::complex<double>* x, const int32_t incx, std::complex<double>* y, const int32_t incy) {
   cblas_zswap(n, x, incx, y, incy);
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

// Rank one update of a matrix
template<typename T>
static void ger(const int32_t m, const int32_t n, const T alpha, const T* x, const int32_t incx, const T* y,
   const int32_t incy, T* a, const int32_t lda);

template<>
void ger(const int32_t m, const int32_t n, const float alpha, const float* x, const int32_t incx, const float* y,
   const int32_t incy, float* a, const int32_t lda) {
   cblas_sger(CBLAS_ORDER::CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}

template<>
void ger(const int32_t m, const int32_t n, const double alpha, const double* x, const int32_t incx, const double* y,
   const int32_t incy, double* a, const int32_t lda) {
   cblas_dger(CBLAS_ORDER::CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}

template<>
void ger(const int32_t m, const int32_t n, const std::complex<float> alpha, const std::complex<float>* x, const int32_t incx, const std::complex<float>* y,
   const int32_t incy, std::complex<float>* a, const int32_t lda) {
   cblas_cgerc(CBLAS_ORDER::CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
}

template<>
void ger(const int32_t m, const int32_t n, const std::complex<double> alpha, const std::complex<double>* x, const int32_t incx, const std::complex<double>* y,
   const int32_t incy, std::complex<double>* a, const int32_t lda) {
   cblas_zgerc(CBLAS_ORDER::CblasColMajor, m, n, &alpha, x, incx, y, incy, a, lda);
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
