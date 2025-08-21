#include <ten/tensor>
#include <ten/io>

int main() {
   using ten::last;
   using ten::seq;
   using ten::mdseq;
   using ten::matrix;
   using ten::smatrix;

   {
      matrix<float> x = ten::ones<ten::matrix<float>>({3, 3});
      matrix<float> y = ten::range<ten::matrix<float>>({3, 3});
      auto a = x > y;
      std::cout << a << std::endl;
      auto agt = x.gt(y);
      std::cout << agt << std::endl;
      auto b = x < y;
      std::cout << b << std::endl;
      auto blt = x.lt(y);
      std::cout << blt << std::endl;
      auto c = x.ge(y);
      std::cout << c << std::endl;
      auto d = x.le(y);
      std::cout << d << std::endl;
      auto e = x.eq(y);
      std::cout << e << std::endl;
   }

   {
      std::cout << "for smatrix\n";
      smatrix<float, 3, 3> x = ten::ones<ten::smatrix<float, 3, 3>>();
      smatrix<float, 3, 3> y = ten::range<ten::smatrix<float, 3, 3>>();
      auto a = x > y;
      std::cout << a << std::endl;
      auto agt = x.gt(y);
      std::cout << agt << std::endl;
      auto b = x < y;
      std::cout << b << std::endl;
      auto blt = x.lt(y);
      std::cout << blt << std::endl;
      auto c = x.ge(y);
      std::cout << c << std::endl;
      auto d = x.le(y);
      std::cout << d << std::endl;
      auto e = x.eq(y);
      std::cout << e << std::endl;
   }
}

