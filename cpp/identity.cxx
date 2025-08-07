#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::identity<ten::matrix<float>>({3, 3});
   std::cout << x << std::endl;

   auto y = ten::identity<ten::smatrix<float, 3, 3>>();
   std::cout << y << std::endl;

   auto z = ten::identity<ten::matrix<float>>({3, 4});
   std::cout << z << std::endl;

}
