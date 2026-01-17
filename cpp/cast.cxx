#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::matrix<float>>({2, 2});
   auto y = ten::cast<double>(x);
   std::cout << y << std::endl;
}
