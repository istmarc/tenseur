#include <ten/tensor>
#include <ten/io>

int main() {
   {
   auto x = ten::range<ten::matrix<float>>({3, 3});
   std::cout << x.shape() << std::endl;
   std::cout << x.strides() << std::endl;

   auto row_0 = x.row(0);
   auto col_0 = x.column(0);

      std::cout << x << std::endl;
      std::cout << row_0 << std::endl;
      std::cout << col_0 << std::endl;
   }

   {
      auto x = ten::range<ten::smatrix<float, 3, 3>>();
      auto row_0 = x.row(0);
      auto col_0 = x.column(0);
      std::cout << row_0 << std::endl;
      std::cout << col_0 << std::endl;
   }

}

