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

// Compute gradient of a tensor
// y = f(x) compute the gradient dy/dx = df/dx (x) in x.grad
template<class Func, class X>
void compute_gradient_unary(X& x) {
   // Unary functions
   if (::ten::is_sqrt<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         x.grad_at(i) = T(1) / (2 * std::sqrt(x[i]));
      }
   }
   if (::ten::is_sqr<Func>::value) {
      for (size_t i = 0; i < input.size(); i++) {
         x.grad_at(i) = T(2) * x[i];
      }
   }
}

// Compute the gradient of a tensor
// z = f(x, y) compute the gradient with respect to x, dz/dx = df/dx (x, y) in x.grad
template<class Func, class Z, class X, class Y>
void compute_gradient_binary(Z& z, X& x, Y& y) {
   if (::ten::is_mul<Func>::value) {
   
   }
}

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
