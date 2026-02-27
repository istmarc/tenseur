#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;
   {
      auto [q, r] = ten::linalg::qr(a);
      std::cout << "Factors" << std::endl;
      std::cout << q << std::endl;
      std::cout << r << std::endl;
   }
}
