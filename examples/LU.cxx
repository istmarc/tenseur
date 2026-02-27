#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;

   auto [p, l, u] = ten::linalg::lu(a);

   std::cout << p << std::endl;
   std::cout << l << std::endl;
   std::cout << u << std::endl;

   std::cout << "PLU" << std::endl;
   matrix<float> x = p * l * u;
   std::cout << x << std::endl;

   std::cout << linalg::det(a) << std::endl;

}
