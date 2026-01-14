#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::matrix<float>>({2, 2});
   std::cout << x << std::endl;

   auto y = x * x;
   auto z = y * x;
   ten::matrix<float> t = z.eval();

   std::cout << t << std::endl;

   float m = ten::max(x).eval().value();
   auto a = m * x;
   std::cout << a.eval() << std::endl;

}
