#include <Ten/Tensor>

int main() {
   using namespace ten;

   auto a = range<Matrix<float>>({2, 3});
   std::cout << a << std::endl;

   auto b = log(a);
   auto c = b.eval();
   std::cout << c << std::endl;


}
