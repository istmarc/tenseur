#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::vector<float>>({10});
   auto y = ten::range<ten::vector<float>>({10});

   ten::axpy(2.0f, x, y);
   std::cout << y << std::endl;

   auto z = 2.0f * x;
   ten::axpy(2.0f, z, y);
   std::cout << y << std::endl;
}
