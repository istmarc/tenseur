#include <ten/tensor>
#include <ten/io>

int main() {
   using ten::last;
   using ten::seq;
   using ten::mdseq;
   using ten::matrix;
   using ten::smatrix;

   {
      auto s = mdseq<2>(seq(0, 1), seq(0, 3));
   }

   {
      matrix<float> x({3, 3});
      auto slice = x(seq(0, 3), seq(0, 1));
      slice = 99.0f;
      std::cout << x << std::endl;
   }

   {
      matrix<float> x({3, 3});
      auto index = mdseq<2>(seq(0, 3), seq(0, 1));
      auto slice = x[index];
      slice = 99.0f;
      std::cout << x << std::endl;
   }

   {
      smatrix<float, 3, 3> x;
      auto slice = x(seq(1, last), seq(0, 1));
      slice = 99.0f;
      std::cout << x << std::endl;
   }

   {
      matrix<float> x({3, 3});
      auto slice = x(seq(0, last), seq(0, 1));
      slice = ten::range<ten::vector<float>>({3});
      std::cout << x << std::endl;
   }

}

