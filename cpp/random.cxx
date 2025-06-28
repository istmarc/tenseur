#include <ten/tensor>

int main(){
   using namespace ten;
   set_seed(1234);

   auto a = rand<matrix<float>>({2, 3});
   std::cout << a << std::endl;

   uniform<> unif;
   std::cout << unif.sample() << std::endl;

   ten::normal norm;
   auto b = norm.sample(1000);
   save(b, "norm.mtx");

   auto x = rand<stensor<float, 2,2>>(1234);
   std::cout << x << std::endl;

   auto y = rand<stensor<float, 2,2>>(1234);
   std::cout << y << std::endl;

}
