#ifndef TENSEUR_KERNELS_BLAS_FUNCTIONS
#define TENSEUR_KERNELS_BLAS_FUNCTIONS

#include <ten/kernels/blas_api.hxx>
#include <ten/types.hxx>

// Level 1 blas funcions
namespace ten::kernels {
/// asum
template <Tensor T> static auto asum(T &&x) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incx = 1;
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template <Column T> static auto asum(T &&x) -> decltype(auto) {
   auto shape = x.shape();
   int32_t n = shape.dim(0);
   int32_t incx = 1;
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template <Row T> static auto asum(T &&x) -> decltype(auto) {
   auto shape = x.shape();
   int32_t n = shape.dim(1);
   int32_t incx = shape.dim(0);
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}

/// axpy
template <typename T, Vector X, Vector Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Vector X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Vector X, Row Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), incy);
}

template <typename T, Column X, Vector Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Column X, Row Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), incy);
}

template <typename T, Column X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Row X, Vector Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), 1);
}

template <typename T, Row X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), 1);
}

template <typename T, Row X, Row Y> static void axpy(const T a, X &&x, Y &y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), incy);
}

// copy
template<Vector X, Vector Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Column Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Row Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), incy);
}

template<Column X, Vector Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template<Column X, Column Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template<Column X, Row Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), incy);
}

template<Row X, Vector Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), 1);
}

template<Row X, Column Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), 1);
}

template<Row X, Row Y>
static void copy(const X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), incy);
}

// iamax
template<Vector X>
static size_t iamax(const X& x) {
   int32_t n = x.size();
   return ::ten::kernels::blas::iamax(n, x.data(), 1);
}

template<Column X>
static size_t iamax(const X& x) {
   int32_t n = x.size();
   return ::ten::kernels::blas::iamax(n, x.data(), 1);
}

template<Row X>
static size_t iamax(const X& x) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::iamax(n, x.data(), incx);
}

// dot
template<Vector X, Vector Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Column Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Row Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), incy);
}

template<Column X, Vector Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template<Column X, Column Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template<Column X, Row Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), incy);
}

template<Row X, Vector Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), 1);
}

template<Row X, Column Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), 1);
}

template<Row X, Row Y>
static auto dot(const X& x, const Y& y) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), incy);
}

// dotc
template<Vector X, Vector Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Column Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Row Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), incy);
}

template<Column X, Vector Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template<Column X, Column Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template<Column X, Row Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), incy);
}

template<Row X, Vector Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), 1);
}

template<Row X, Column Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), 1);
}

template<Row X, Row Y>
static auto dotc(const X& x, const Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   int32_t incy = y.shape().dim(0);
   return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), incy);
}

// nrm2
template<Vector X>
static auto nrm2(const X& x) {
   int32_t n = x.size();
   return ::ten::kernels::blas::nrm2(n, x.data(), 1);
}

template<Column X>
static auto nrm2(const X& x) {
   int32_t n = x.size();
   return ::ten::kernels::blas::nrm2(n, x.data(), 1);
}

template<Row X>
static auto nrm2(const X& x) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   return ::ten::kernels::blas::nrm2(n, x.data(), incx);
}

// scal
template<typename T, Vector X>
static void scal(const T alpha, X& x) {
   int32_t n = x.size();
   ::ten::kernels::blas::scal(n, alpha, x.data(), 1);
}

template<typename T, Column X>
static void scal(const T alpha, X& x) {
   int32_t n = x.size();
   ::ten::kernels::blas::scal(n, alpha, x.data(), 1);
}

template<typename T, Row X>
static void scal(const T alpha, X& x) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::scal(n, alpha, x.data(), incx);
}

// swap
template<Vector X, Vector Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Column Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template<Vector X, Row Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), incy);
}

template<Column X, Vector Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template<Column X, Column Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template<Column X, Row Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   int32_t incy = y.shape().dim(0);
   ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), incy);
}

template<Row X, Vector Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), 1);
}

template<Row X, Column Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), 1);
}

template<Row X, Row Y>
static void swap(X& x, Y& y) {
   int32_t n = x.size();
   int32_t incx = x.shape().dim(0);
   int32_t incy = y.shape().dim(1);
   ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), incy);
}

// gemv
// Matrix vector multiplication
template <typename T, Matrix A, Vector B, Vector C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Vector B, Column C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Vector B, Row C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = c.shape().dim(0);
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Column B, Vector C>
static void gemv(const T alpha, const A& a, const B &b,const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Column B, Row C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = c.shape().dim(0);
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Row B, Vector C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = b.shape().dim(0);
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Row B, Column C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = b.shape().dim(0);
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

template <typename T, Matrix A, Row B, Row C>
static void gemv(const T alpha, const A& a, const B &b, const T beta, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = b.shape().dim(0);
   const size_t incc = c.shape().dim(0);
   ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb, beta,
              c.data(), incc);
}

} // namespace ten::kernels

#endif
