#include <ten/tensor>

int main() {
   auto x = ten::range<ten::matrix<float>>({2, 2});
   std::cout << x << std::endl;

   auto y = x * x;
   auto z = y * x;
   ten::matrix<float> t = z.eval();

   std::cout << t << std::endl;
   //ten::matrix<float> z = 

   auto a = ten::max(x) * x;
   std::cout << a.eval() << std::endl;

}
