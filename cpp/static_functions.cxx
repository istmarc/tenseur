#include <ten/io>
#include <ten/tensor>

int main() {
   ten::set_seed(123);

   {
      std::cout << "MSE loss\n";
      auto x = ten::range<ten::svector<float, 10>>(1.);
      auto y = ten::range<ten::svector<float, 10>>(2.);
      auto loss = ten::mse(x, y);
      loss.eval();
      std::cout << loss.value() << std::endl;
   }

   {
      std::cout << "Sigmoid\n";
      auto x = ten::range<ten::svector<float, 10>>(1.);
      auto y = ten::sigmoid(x);
      y.eval();
      std::cout << y.value() << std::endl;
   }

   {
      std::cout << "Leaky relu\n";
      auto x = ten::rand_norm<ten::svector<float, 10>>();
      auto y = ten::leaky_relu(x);
      y.eval();
      std::cout << y.value() << std::endl;
   }

   {
      std::cout << "Relu\n";
      auto x = ten::rand_norm<ten::svector<float, 10>>();
      auto y = ten::relu(x);
      y.eval();
      std::cout << y.value() << std::endl;
   }

   {
      std::cout << "scalar * matrix\n";
      auto x = ten::range<ten::smatrix<float, 3, 3>>(1.);
      auto y = 2.0f * x;
      std::cout << y.eval() << std::endl;
   }

   {
      std::cout << "scalar * tensor\n";
      auto x = ten::range<ten::stensor<float, 3, 3, 3>>(1.);
      auto y = 2.0f * x;
      auto z = y.eval();
      std::cout << decltype(z)::shape_type() << std::endl;
   }

   {
      std::cout << "smatrix * svector\n";
      auto x = ten::range<ten::smatrix<float, 2, 3>>(1.);
      auto y = ten::range<ten::svector<float, 3>>(1.);
      auto z = x * y;
      std::cout << z.eval() << std::endl;
   }

   {
      std::cout << "smatrix * smatrix\n";
      auto x = ten::range<ten::smatrix<float, 2, 3>>(1.);
      auto y = ten::range<ten::smatrix<float, 3, 2>>(1.);
      auto z = x * y;
      std::cout << z.eval() << std::endl;
   }

   {
      std::cout << "svector * svector\n";
      auto x = ten::range<ten::svector<float, 10>>(1.);
      auto y = ten::range<ten::svector<float, 10>>(1.);
      auto z = x * y;
      std::cout << z.eval() << std::endl;
   }

   {
      std::cout << "tensor * scalar\n";
      auto x = ten::range<ten::stensor<float, 3, 3, 3>>(1.);
      auto y = x * 2.0f;
      auto z = y.eval();
      std::cout << decltype(z)::shape_type() << std::endl;
   }
}
