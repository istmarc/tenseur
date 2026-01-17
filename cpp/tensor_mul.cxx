#include <ten/tensor>
#include <ten/io>

int main() {
   ten::tensor<float, 3> x({2, 3, 4});
   ten::tensor<float, 3> y({2, 3, 4});
   auto z = ten::flatten(x) * ten::flatten(y);
   z.eval();
   std::cout << z.value() << std::endl;
}
