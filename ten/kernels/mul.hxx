#ifndef TA_KERNELS_MUL_HXX
#define TA_KERNELS_MUL_HXX

#include <ten/types.hxx>

namespace ten::kernels {
// matrix * vector
template <class A, class B, class C>
void mul(const A &a, const B &b, C &c)
   requires ::ten::is_matrix_node<A>::value && ::ten::is_vector_node<B>::value
{
   size_t m = a.dim(0);
   size_t n = a.dim(1);
   using blas::transop;
   using T = typename A::value_type;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : n);
   const size_t incb = 1;
   const size_t incc = 1;
   blas::gemv(transa, m, n, T(1.), a.data(), lda, b.data(), incb, T(0.),
              c.data(), incc);
}

// Multiply two dense matrices
// C <- A * B
template <class A, class B, class C>
void mul(const A &a, const B &b, C &c)
   requires ::ten::is_matrix_node<A>::value && ::ten::is_matrix_node<B>::value &&
            ::ten::is_matrix_node<C>::value
{
   size_t m = a.dim(0);
   size_t k = a.dim(1);
   size_t n = b.dim(1);
   using blas::transop;
   using T = typename A::value_type;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const transop transb = (b.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : k);
   const size_t ldb = (transa == transop::no ? k : n);
   blas::gemm(transa, transb, m, n, k, T(1.), a.data(), lda, b.data(), ldb,
              T(0.), c.data(), m);
}


// Multiply and add two dense matrices
// C <- alpha * A * B + beta * C
template <class A, class B, class C, class T>
void mul_add(const A &a, const B &b, C &c, const T& alpha, const T& beta)
   requires ::ten::is_matrix_node<A>::value && ::ten::is_matrix_node<B>::value &&
            ::ten::is_matrix_node<C>::value
{
   size_t m = a.dim(0);
   size_t k = a.dim(1);
   size_t n = b.dim(1);
   using blas::transop;
   const transop transa = (a.is_transposed() ? transop::trans : transop::no);
   const transop transb = (b.is_transposed() ? transop::trans : transop::no);
   const size_t lda = (transa == transop::no ? m : k);
   const size_t ldb = (transa == transop::no ? k : n);
   blas::gemm(transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb,
              beta, c.data(), m);
}

} // namespace ten::kernels

#endif
