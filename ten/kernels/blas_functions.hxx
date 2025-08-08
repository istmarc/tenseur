#ifndef TENSEUR_KERNELS_BLAS_FUNCTIONS
#define TENSEUR_KERNELS_BLAS_FUNCTIONS

#include <ten/types.hxx>
#include <ten/kernels/blas_api.hxx>

// Level 1 blas funcions
namespace ten::kernels{

template<Tensor T>
static auto asum(T&& x) -> decltype(auto) {
   int32_t n = x.size();
   int32_t incx = 1;
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template<Column T>
static auto asum(T&& x) -> decltype(auto) {
   auto shape = x.shape();
   int32_t n = shape.dim(0);
   int32_t incx = 1;
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template<Row T>
static auto asum(T&& x) -> decltype(auto) {
   auto shape = x.shape();
   int32_t n = shape.dim(1);
   int32_t incx = shape.dim(0);
   return ::ten::kernels::blas::asum(n, x.data(), incx);
}


}


#endif
