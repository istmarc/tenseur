#ifndef TENSEUR_AUTOGRAD
#define TENSEUR_AUTOGRAD

#include <ten/types.hxx>
#include <ten/functional_types.hxx>

namespace ten{

// TODO Compute scalar gradient
/*
template<class Func, class T>
void compute_scalar_gradient(ten::scalar<T>& input) {
   if (::ten::is_sqrt<Func>::value) {
      input.set_grad(T(1) / (2 * std::sqrt(input.value())));
   }
}*/

// Compute gradient of a dynamic tensor
template<class Func, class T, class Shape, storage_order order, class Storage, class Allocator>
void compute_gradient(ten::ranked_tensor<T, Shape, order, Storage, Allocator>& input) {
   // Unary functions
   if (::ten::is_sqrt<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         input.grad_at(i) = T(1) / (2 * std::sqrt(input[i]));
      }
   }
   if (::ten::is_sqr<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         input.grad_at(i) = T(2) * input[i];
      }
   }
}

// TODO Compute gradient of a static tensor
/*template<class Func, class T, size_t ...Dims>
void compute_gradient(ten::stensor<T, Dims...>& input) {
}*/

// Compute gradient of a dynamic tensor using the chain rule
template<class Func, class T, class Shape, storage_order order, class Storage, class Allocator>
void compute_gradient_chain_rule(ten::ranked_tensor<T, Shape, order, Storage, Allocator>& input,
   ten::ranked_tensor<T, Shape, order, Storage, Allocator>& grad) {
   // Unary functions
   if (::ten::is_sqrt<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         input.grad_at(i) = grad[i] * T(1) / (2 * std::sqrt(input[i]));
      }
   }
   if (::ten::is_sqr<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         input.grad_at(i) = grad[i] * T(2) * input[i];
      }
   }
}

}

#endif
