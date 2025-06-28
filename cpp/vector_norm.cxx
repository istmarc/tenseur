#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto v = ten::range<ten::vector<float>>({3}, 1.);
   std::cout << v << std::endl;

   std::cout << "Norms" << std::endl;
   std::cout << ten::norm(v, ten::vector_norm::l1) << std::endl;
   std::cout << ten::norm(v, ten::vector_norm::l2) << std::endl;
   std::cout << ten::norm(v, 3) << std::endl;
   std::cout << ten::norm(v, 4) << std::endl;
   std::cout << ten::norm(v, ten::vector_norm::linf) << std::endl;
}
