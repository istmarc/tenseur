#ifndef TEN_KERNELS_BLAS_API_HXX
#define TEN_KERNELS_BLAS_API_HXX

#include <cstddef>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <lapack/lapack.h>
#endif

namespace ten::kernels::lapack {

enum class transop : char { no = 'N', trans = 'T' };

static LAPACK_TRANSPOSE cast(const transop op) {
   if (op == transop::no) {
      return LAPACK_TRANSPOSE::LapackNoTrans;
   } else {
      return LAPACK_TRANSPOSE::LapackTrans;
   }
}

/// QR factorization
template <typename T>
static void qrFactorization();

template<>
static void qrFactorization() {

};

} // namespace ten::kernels::blas

#endif
