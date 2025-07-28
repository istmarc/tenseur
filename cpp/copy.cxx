#include <ten/tensor>
#include <ten/io>

int main() {
   auto a = ten::range<ten::matrix<float>>({2, 3});
   auto b = a.copy();
   std::cout << a << std::endl;
   std::cout << b << std::endl;

   auto x = ten::range<ten::smatrix<float, 2, 3>>();
   auto y = x.copy();
   std::cout << x << std::endl;
   std::cout << y << std::endl;

}
