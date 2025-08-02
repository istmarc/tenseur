#include <ten/tensor>
#include <ten/io>

int main() {
   ten::verbose(true);

   ten::matrix<float> x = ten::range<ten::matrix<float>>({3, 3});
   std::cout << "x = " << x << std::endl;

   x = ten::sqrt(x);
   std::cout << x << std::endl;

}
