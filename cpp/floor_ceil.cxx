#include <ten/tensor>

int main() {
   auto x = ten::rand<float, 2>({3, 3});
   std::cout << x << std::endl;
   std::cout << ten::floor(x).eval() << std::endl;
   std::cout << ten::ceil(x).eval() << std::endl;

}
