#include <ten/tensor.hxx>
#include <bitset>

int main() {
   using namespace ten;

   matrix<float> a({2, 2}, {1., 2., 2., 1.});
   std::cout << a << std::endl;
   auto b = symmetric(a);
   std::cout << "a" << std::endl;
   std::cout << std::boolalpha << a.is_symmetric() << std::endl;
   std::cout << std::boolalpha << a.is_transposed() << std::endl;
   std::cout << std::boolalpha << a.is_hermitian() << std::endl;
   std::cout << std::boolalpha << a.is_lower_tr() << std::endl;
   std::cout << std::boolalpha << a.is_upper_tr() << std::endl;

   std::cout << "b" << std::endl;
   std::cout << std::boolalpha << b.is_symmetric() << std::endl;
   std::cout << std::boolalpha << b.is_transposed() << std::endl;
   std::cout << std::boolalpha << b.is_hermitian() << std::endl;
   std::cout << std::boolalpha << b.is_lower_tr() << std::endl;
   std::cout << std::boolalpha << b.is_upper_tr() << std::endl;

   std::vector<int16_t> x({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096});
   for (auto xx : x){
      std::cout << std::bitset<16>(xx) << std::endl;
   }

   //smatrix<float, 2, 2> x({1., 2., 2., 1.});
   //std::cout << x << std::endl;
   //auto y = symmetric(x);
   //std::cout << std::boolalpha << y.is_symmetric() << std::endl;


}
