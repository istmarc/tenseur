#include <ten/tensor>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 3});
   auto b = ten::range<ten::matrix<float>>({3, 3});
   auto c = ten::range<ten::matrix<float>>({3, 3});

   ten::gemm(2.0f, a, b, 1.0f, c);

   std::cout << c << std::endl;

   auto z = a * b;
   ten::gemm(2.0f, a, z, 1.0f, c);
   std::cout << c << std::endl;

   ten::gemm(2.0f, ten::range<ten::matrix<float>>({3, 3}), z, 1.0f, c);
   std::cout << c << std::endl;
}
