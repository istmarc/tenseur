#include "ten/tensor.hxx"
#include <ten/tensor>

int main() {
   auto x = ten::range<ten::matrix<float>>({2, 3});
   std::cout << std::boolalpha;
   std::cout << x.is_transposed() << std::endl;
   std::cout << x.is_symmetric() << std::endl;
   std::cout << x.is_hermitian() << std::endl;
   std::cout << x.is_lower_tr() << std::endl;
   std::cout << x.is_upper_tr() << std::endl;
   std::cout << "---\n";

   auto y = ten::transposed(x);
   std::cout << std::boolalpha;

   std::cout << y.is_transposed() << std::endl;
   std::cout << y.is_symmetric() << std::endl;
   std::cout << y.is_hermitian() << std::endl;
   std::cout << y.is_lower_tr() << std::endl;
   std::cout << y.is_upper_tr() << std::endl;

}
