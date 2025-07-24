#include <ten/tensor>

int main(){
   using namespace ten;
   set_seed(1234);

   auto a = rand_norm<matrix<float>>({2, 3});
   std::cout << a << std::endl;

   set_seed(1234);
   auto x = rand_norm<stensor<float, 2,2>>();
   std::cout << x << std::endl;

   auto y = rand_norm<stensor<float, 2,2>>();
   std::cout << y << std::endl;

}
