#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
   using namespace ten;
   auto a = ten::range<matrix<float>>({4, 4});
   std::cout << a << std::endl;
   ten::linalg::qr_fact<float> qrfact;
   qrfact.factorize(a);
   auto q = qrfact.q();
   auto r = qrfact.r();

   std::cout << q << std::endl;
   std::cout << r << std::endl;

   std::cout << (q * r).eval() << std::endl;

   std::cout << a << std::endl;

   std::cout << "Factors\n";
   auto [qq, rr] = qrfact.factors();
   std::cout << qq << std::endl;
   std::cout << rr << std::endl;

   {
      auto [q, r] = ten::linalg::qr(a);
      std::cout << "Factors" << std::endl;
      std::cout << q << std::endl;
      std::cout << r << std::endl;
   }
}
