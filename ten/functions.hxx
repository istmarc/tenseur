#ifndef TENSEUR_FUNCTIONS
#define TENSEUR_FUNCTIONS

#include <ten/types.hxx>

namespace ten{

// Gemm
template<class T, Expr X, Expr Y, Tensor C>
requires(std::is_same_v<T, typename C::value_type>)
static void gemm(const T /*alpha*/, X && /*x*/, Y&& /*y*/, const T /*beta*/, C& /*c*/);

}



#endif
