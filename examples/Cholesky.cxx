#include <ios>
#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});
   std::cout << a << std::endl;

   auto [l, u] = ten::linalg::cholesky(a);

   std::cout << l << std::endl;
   std::cout << u << std::endl;

   ten::matrix<float> lu = l*u;
   ten::matrix<float> v = lu - a;
   std::cout << std::boolalpha << ten::all_close(v, 1e-3) << std::endl;

}
