#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});
   std::cout << a << std::endl;

   auto [u, s, vt] = ten::linalg::svd(a);

   std::cout << u << std::endl;
   std::cout << s << std::endl;
   std::cout << vt << std::endl;

   auto sigma = ten::zeros<ten::matrix<float>>({3, 3});
   for (size_t i = 0; i < 3; i++) {
      sigma(i,i) = s[i];
   }

   std::cout << sigma << std::endl;

   auto y = (u * sigma).eval();
   auto z = y * vt;
   std::cout << z.eval() << std::endl;

   //::ten::matrix<float> v = (l*u).eval() - a;
   //std::cout << ten::all_close(v) << std::endl;

}
