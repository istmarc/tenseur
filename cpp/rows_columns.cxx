#include <ten/tensor>

template <class T> void print(const T &t, size_t n) {
   for (size_t i = 0; i < n; i++) {
      std::cout << t[i] << " ";
   }
   std::cout << std::endl;
};

int main() {
   auto x = ten::range<ten::matrix<float>>({3, 4});
   std::cout << x << std::endl;

   {
      auto col = x.column(2);
      // should print 6, 7, 8
      print(col, 3);
   }

   {
      auto row = x.row(2);
      // should print 2, 5, 8, 11
      print(row, 4);
   }
}
