#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto v = ten::range<ten::vector<float>>({3}, 1.);
   auto w = ten::range<ten::vector<float>>({3}, 1.);
   std::cout << v << std::endl;

   std::cout << "Dot product" << std::endl;
   std::cout << ten::dot(v, w) << std::endl;
   std::cout << "Outer product" << std::endl;
   std::cout << ten::outer(v, w) << std::endl;

}
