#include <ten/tensor>
#include <ten/io>

int main() {
   ten::verbose(true);

   ten::matrix<float> a = ten::range<ten::matrix<float>>({3, 3});
   ten::matrix<float> b = ten::range<ten::matrix<float>>({3, 3});
   ten::matrix<float> c = ten::range<ten::matrix<float>>({3, 3});

   {
      //a = ten::sqrt(a);
      //std::cout << a << std::endl;
   }

   {
      auto d = a * b;
      std::cout << d.eval() << std::endl;
   }

  /* {
      std::cout << "D\n";
      auto d = a * b + c;
      std::cout << d << std::endl;
      std::cout << d.eval() << std::endl;
   }*/

   {
      std::cout << "c = \n";
      std::cout << c << std::endl;
      c = a * b + c;
      std::cout << "After call to gemm\n";
      std::cout << c << std::endl;
   }

}
