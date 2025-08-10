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

template<class X, class Y>
auto dot(const X& x, const Y& y) {
   using value_type = typename X::value_type;
   value_type d = 0.;
   for (size_t i = 0;i < x.size(); i++) {
      d += x[i] * y[i];
   }
   return d;
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

   {
      std::cout << "axpy with column, column\n";
      auto a = ten::range<ten::matrix<float>>({3,3});
      auto col_0 = a.column(0);
      auto col_1 = a.column(1);
      ten::axpy(1.0f, col_0, col_1);
      std::cout << a << std::endl;
   }

   {
      std::cout << "axpy with row, row\n";
      auto a = ten::range<ten::matrix<float>>({3,3});
      auto row_0 = a.row(0);
      auto row_1 = a.row(1);
      ten::axpy(1.0f, row_0, row_1);
      std::cout << a << std::endl;
   }

   {
      std::cout << "ten::copy vectors\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto row_0 = a.row(0);
      auto row_1 = a.row(1);
      ten::copy(row_0, row_1);
      std::cout << a << std::endl;
      auto col_0 = a.column(0);
      auto col_1 = a.column(1);
      ten::copy(col_0, col_1);
      std::cout << a << std::endl;
   }

   {
      std::cout << "iamax\n";
      auto a = ten::rand_norm<ten::matrix<float>>({3, 3});
      std::cout << a << std::endl;
      std::cout << ten::iamax(a.row(0)) << std::endl;
      std::cout << ten::iamax(a.column(0)) << std::endl;
   }

   {
      std::cout << "dot product\n";
      auto a = ten::rand_norm<ten::vector<float>>({3});
      auto b = ten::rand_norm<ten::vector<float>>({3});
      std::cout << a << std::endl;
      std::cout << b << std::endl;
      std::cout << ten::dot(a, b) << std::endl;
      std::cout << dot(a, b) << std::endl;
   }

   {
      std::cout << "conjugated dot product\n";
      ten::vector<std::complex<float>> a({2}, {1.0f+2.0fi, 3.0f-1.0fi});
      ten::vector<std::complex<float>> b({2}, {2.0f+1.0fi, 3.0f-2.0fi});
      std::cout << a << std::endl;
      std::cout << b << std::endl;
      std::cout << "a*.b = " << ten::dotc(a, b) << std::endl;
   }

   {
      std::cout << "nrm2\n";
      auto a = ten::range<ten::vector<float>>({3}, 1.);
      std::cout << a << std::endl;
      std::cout << "nrm2(a) = " << ten::nrm2(a) << std::endl;
   }

}
