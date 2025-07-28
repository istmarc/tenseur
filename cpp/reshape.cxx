#include <ten/tensor>

int main() {

   {
      auto x = ten::range<ten::smatrix<float, 3, 3>>();
      auto y = ten::reshape<ten::shape<9>>(x);
      std::cout << y.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::smatrix<float, 3, 3>>();
      auto y = ten::reshape<9>(x);
      std::cout << y.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::matrix<float>>({3, 3});
      auto y = ten::reshape<1>(x, {9});
      std::cout << y.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::smatrix<float, 3, 3>>();
      auto y = ten::transpose(x);
      std::cout << y.eval() << std::endl;
   }

}
