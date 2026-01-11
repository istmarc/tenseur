#ifndef TENSEUR_FUNCTIONS
#define TENSEUR_FUNCTIONS

#include <ten/types.hxx>

namespace ten {

// Blas functions
/// gemm
template <class T, Expr X, Expr Y, Tensor C>
   requires(std::is_same_v<T, typename C::value_type>)
static void gemm(const T /*alpha*/, X && /*x*/, Y && /*y*/, const T /*beta*/,
                 C & /*c*/);

// Like
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
ten::ranked_tensor<T, Shape, order, Storage, Allocator>
like(const ten::ranked_tensor<T, Shape, order, Storage, Allocator> & /*x*/);

} // namespace ten

#endif
