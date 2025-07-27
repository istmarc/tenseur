#include <ten/tensor>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 4});

   std::cout << a << std::endl;
   {
      auto col_0 = a.column(0);
      std::cout << col_0 << std::endl;
   }

   {
      auto row_0 = a.row(0);
      std::cout << row_0 << std::endl;
   }

   {
      auto b = ten::range<ten::vector<float>>({4});
      auto c = a * b;
      std::cout << c.eval() << std::endl;
      auto col_0 = a.column(0);
      col_0 = c;
      std::cout << a << std::endl;
   }

   {
      auto row_0 = a.row(0);
      auto x = ten::range<ten::vector<float>>({4});
      std::cout << "x = " << x << std::endl;
      row_0 = x;
      auto row_1 = a.row(1);
      row_1 = x;
      std::cout << a << std::endl;
   }

   {
      auto col_0 = a.column(0);
      auto x = ten::range<ten::vector<float>>({3});
      col_0 = x;
      std::cout << a << std::endl;
      col_0 = x;
      auto col_2 = a.column(2);
      col_2 = x;
      std::cout << a << std::endl;
   }

}
