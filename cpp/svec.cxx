#include <ten/tensor.hxx>

int main() {
   using namespace ten;

   svector<float, 1'000'000> a;
   std::cout << a.size() << std::endl;
   std::cout << a << std::endl;

}
