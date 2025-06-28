#include <ten/tensor>

int main() {
   ten::svector<float, 10> a;
   ten::svector<float, 10> b;

   auto c = a + b;

   std::cout << c.eval() << std::endl;
}
