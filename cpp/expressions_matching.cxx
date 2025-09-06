#include <ten/tensor>
#include <ten/io>

int main() {
   ten::verbose(true);

   ten::matrix<float> a = ten::range<ten::matrix<float>>({3, 3});
   ten::matrix<float> b = ten::range<ten::matrix<float>>({3, 3});
   ten::matrix<float> c = ten::range<ten::matrix<float>>({3, 3});

   {
      auto d = a * b;
      std::cout << d.eval() << std::endl;
   }

   {
      std::cout << "c = \n";
      std::cout << c << std::endl;
      c = a * b + c;
      std::cout << "After call to gemm\n";
      std::cout << c << std::endl;
   }

   {
      /*auto d = 2.0f * (a * b);
      std::cout << "d = \n";
      std::cout << d.eval() << std::endl;
      */
      c = 2.0f * a * b + c;
      std::cout << "After call to gemm\n";
      std::cout << c << std::endl;
   }

}
