#include "ten/types.hxx"
#include <ten/tensor>
#include <ten/io>

template<typename T>
requires (ten::is_tensor_v<T> || ten::is_column_v<T> || ten::is_row_v<T>)
auto absolute_sum(const T& x) -> decltype(auto) {
   using value_type = T::value_type;
   value_type s = 0.;
   for(size_t i = 0; i < x.size(); i++) {
      s += std::abs(x[i]);
   }
   return s;
}

int main() {

   {
      std::cout << "For a vector\n";
      auto a = ten::rand_norm<ten::vector<float>>({10});
      std::cout << a << std::endl;
      auto b = ten::asum(a);
      std::cout << b << std::endl;
      std::cout << absolute_sum(a) << std::endl;
   }

   {
      std::cout << "For a matrix row and column\n";
      auto a = ten::rand_norm<ten::matrix<float>>({3, 3});
      std::cout << a << std::endl;
      auto b = a.column(0);
      std::cout << "Column = " << b << std::endl;
      std::cout << ten::asum(b) << std::endl;
      std::cout << absolute_sum(b) << std::endl;
      auto c = a.row(0);
      std::cout << "Row = " << c << std::endl;
      std::cout << ten::asum(c) << std::endl;
      std::cout << absolute_sum(c) << std::endl;
   }
}
