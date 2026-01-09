#ifndef TENSEUR_AUTOGRAD
#define TENSEUR_AUTOGRAD

#include <ten/types.hxx>
#include <ten/functional_types.hxx>

namespace ten{

// Compute scalar gradient
template<class Func, class T>
void compute_gradient_scalar(ten::scalar<T>& input, Func& f) {
   input.grad_value() = f.gradient(input.value());
}

// Compute gradient of a tensor
// y = f(x) compute the gradient dy/dx = df/dx (x) in x.grad
template<class Func, class X>
void compute_gradient_unary(X& x, Func &f) {
   f.gradient(x, x.grad());
}

// Compute the gradient of a tensor
// z = f(x, y) compute the gradient with respect to x, dz/dx = df/dx (x, y) in x.grad
template<class Func, class Z, class X, class Y>
void compute_gradient_binary(Z& z, X& x, Y& y, Func& f) {
   f.gradient_left(x, y, z);
   f.gradient_right(x, y, z);
}

// Compute gradient of a dynamic tensor using the chain rule
// gradZ = gradX * gradY
template<class Func, class Input, class GradX, class GradY>
void compute_gradient_chain_rule(Input& input, GradX& gradx, GradY& grady) {
   for (size_t i = 0; i < input.size(); i++) {
      input.grad_at(i) = gradx[i] * grady[i];
   }
}

template<class GradientF, class GradientG>
void backward_chain(GradientF& gradf, GradientG& gradg) {
   for (size_t i = 0; i< gradf.size(); i++) {
      gradf[i] *= gradg[i];
   }
}

}

#endif
