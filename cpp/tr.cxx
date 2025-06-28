#include <ten/tensor>

int main() {
   using namespace ten;

   auto a = range<matrix<float>>({2, 3});
   std::cout << a << std::endl;

   auto b = transpose(a);
   auto c = b.eval();
   std::cout << c << std::endl;

   auto e = range<smatrix<float, 2, 3>>();
   std::cout << e << std::endl;
   auto f = transpose(e);
   auto g = f.eval();
   std::cout << g << std::endl;

}
