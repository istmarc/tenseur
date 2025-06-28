#include <ten/tensor.hxx>

int main() {
   using namespace ten;

   auto a = linear<smatrix<float, 2, 5>>(0., 1.);
   std::cout << a << std::endl;

   diagonal<float> b(2 , 2);
   for(size_t i = 0; i < 2; i++) {
      b[i] = i + 1;
   }

   std::cout << b << std::endl;

   std::cout << b.shape() << std::endl;
   std::cout << b.strides() << std::endl;
   auto s = b.storage().get()->data();
   std::cout << s[0] << std::endl;
   std::cout << s[1] << std::endl;

}
