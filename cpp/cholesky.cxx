#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});
   std::cout << a << std::endl;

   cholesky cholesky_fact;
   cholesky_fact.factorize(a);

   auto l = cholesky_fact.l();
   auto u = cholesky_fact.u();

   std::cout << l << std::endl;
   std::cout << u << std::endl;

   std::cout << "LU" << std::endl;
   std::cout << (l * u).eval() << std::endl;

   ::ten::matrix<float> v = (l*u).eval() - a;
   std::cout << ten::all_close(v) << std::endl;

}
