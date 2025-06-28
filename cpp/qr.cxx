#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;
   qr qr_fact;
   qr_fact.factorize(a);
   auto q = qr_fact.q();
   auto r = qr_fact.r();

   std::cout << q << std::endl;
   std::cout << r << std::endl;

   std::cout << (q * r).eval() << std::endl;

   std::cout << a << std::endl;

   std::cout << "Factors\n";
   auto [qq, rr] = qr_fact.factors();
   std::cout << qq << std::endl;
   std::cout << rr << std::endl;
}
