#include <ten/tensor>
#include <ten/io>

int main() {
   ten::stensor<float, 10> x;
   std::cout << x << std::endl;

   auto grad = x.grad_node();
   for (size_t i = 0; i < 10;i++) {
      std::cout << x.grad_at(i) << std::endl;
   }

}

