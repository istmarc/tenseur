#include <ten/tensor>

namespace ten::nn {

// Relu
struct relu {

   template <Expr ExprType> auto operator()(ExprType &x) {
      return ten::relu(x);
   }
};

// Leaky relu
struct leaky_relu {

   template <Expr ExprType> auto operator()(ExprType &x) {
      return ten::leaky_relu(x);
   }
};

// Sigmoid
struct sigmoid {

   template <Expr ExprType> auto operator()(ExprType &x) {
      return ten::sigmoid(x);
   }
};

} // namespace ten::nn
