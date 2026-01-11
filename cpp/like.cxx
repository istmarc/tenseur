#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::matrix<float>>({3, 3});
   auto y = ten::like(x);
   std::cout << y << std::endl;
}
