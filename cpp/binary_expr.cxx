#include <ten/tensor>

int main(){
   auto x = ten::range<ten::vector<float>>({10}, 1.);
   auto y = ten::range<ten::vector<float>>({10}, 1.);

   {
      auto z = x + y;
      auto t = z.eval();
      std::cout << t << std::endl;
   }

   {
      auto z = (x - y).eval();
      std::cout << z << std::endl;
   }

   {
      auto z = (x / y).eval();
      std::cout << z << std::endl;
   }


   {
      std::cout << "X\n";
      auto z = (x * y).eval();
      std::cout << z << std::endl;
   }

   {
      std::cout << "Matrix multiplication\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto b = ten::range<ten::matrix<float>>({3, 3});
      auto c = (a * b).eval();
      std::cout << c << std::endl;
   }

   {
      std::cout << "Matrix vector multiplication\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto b = ten::range<ten::vector<float>>({3});
      auto c = (a * b).eval();
      std::cout << c << std::endl;
   }

}
