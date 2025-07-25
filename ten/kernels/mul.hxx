#ifndef TA_KERNELS_MUL_HXX
#define TA_KERNELS_MUL_HXX

#include <ten/types.hxx>

namespace ten::kernels {
/// Matrix vector multiplication
template <Matrix A, Vector B, Vector C>
void mul(A&& a, B &&b, C &c)
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   using T = typename A::value_type;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = 1;
   ::ten::kernels::blas::gemv(transa, m, n, T(1.), a.data(), lda, b.data(), incb, T(0.),
              c.data(), incc);
}

/// Multiply two dense matrices
/// C <- A * B
template <Matrix A, Matrix B, Matrix C>
void mul(A &&a, B &&b, C &c)
{
   size_t m = a.dim(0);
   size_t k = a.dim(1);
   size_t n = b.dim(1);
   using blas::transop;
   using T = typename std::remove_cvref_t<A>::value_type;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const transop transb = (b.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : k);
   const size_t ldb = (transa == transop::no ? k : n);
   ::ten::kernels::blas::gemm(transa, transb, m, n, k, T(1.), a.data(), lda, b.data(), ldb,
              T(0.), c.data(), m);
}

/// Multiply and add two dense matrices
/// C <- alpha * A * B + beta * C
template <Matrix A, Matrix B, Matrix C, class T>
void mul_add(A &&a, B &&b, C &c, const T &alpha, const T &beta) {
   size_t m = a.dim(0);
   size_t k = a.dim(1);
   size_t n = b.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const transop transb = (b.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : k);
   const size_t ldb = (transa == transop::no ? k : n);
   ::ten::kernels::blas::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb,
              beta, c.data(), m);
}

/// scale and add a vector
template <Vector X, Vector Y, class T> void axpy(const T a, X &&x, Y &y) {
   size_t n = x.size();
   ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

} // namespace ten::kernels

#endif
