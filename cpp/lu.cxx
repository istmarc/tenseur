#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;

   lu lu_fact;
   lu_fact.factorize(a);

   auto p = lu_fact.p();
   std::cout << p << std::endl;

   auto l = lu_fact.l();
   auto u = lu_fact.u();

   std::cout << l << std::endl;
   std::cout << u << std::endl;

   std::cout << "LU" << std::endl;
   std::cout << (l * u).eval() << std::endl;

   std::cout << "PLU" << std::endl;
   matrix<float> x = (p * l).eval() * u;
   std::cout << x << std::endl;

   std::cout << det(a) << std::endl;

}
