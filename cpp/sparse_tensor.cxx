#include <ten/tensor>

int main() {
   {
      ten::sparse_tensor<float, 2> x({3, 3});
      x(0,0) = 1.;
      x(1, 1) = 1.;
      auto storage = x.storage();
   }

   {
      ten::coo_matrix<float> x({3, 3});//, {{0,0}, {1, 1}}, {1.0f, 2.0f});
      std::cout << x(0, 0) << std::endl;
      std::cout << x(1, 1) << std::endl;
   }



}
