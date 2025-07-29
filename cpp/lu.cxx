#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;

   auto [p, l, u] = ten::linalg::lu(a);

   std::cout << l << std::endl;
   std::cout << u << std::endl;

   std::cout << "LU" << std::endl;
   std::cout << (l * u).eval() << std::endl;

   std::cout << "PLU" << std::endl;
   // FIXME bad optional access
   matrix<float> x = (p * l).eval() * u;
   std::cout << x << std::endl;

   std::cout << linalg::det(a) << std::endl;

}
