#include <ten/tensor>
#include <ten/linalg>

int main() {
   ten::matrix<float> x({3, 3}, {1., 0., 1., 0., -1., -2., 2., 1., 0.});

   std::cout << x << std::endl;
   auto y = ten::inv(x);
   std::cout << y << std::endl;

   std::cout << (x * y).eval() << std::endl;
}
