#include <ten/tensor>

int main(){
   auto x = ten::range<ten::vector<float>>({10}, 1.);
   auto y = ten::range<ten::vector<float>>({10}, 1.);

   {
      auto z = x + y;
      auto t = z.eval();
      std::cout << t << std::endl;
   }

   {
      auto z = (x - y).eval();
      std::cout << z << std::endl;
   }

   {
      auto z = (x / y).eval();
      std::cout << z << std::endl;
   }


   {
      std::cout << "X\n";
      auto z = (x * y).eval();
      std::cout << z << std::endl;
   }

   {
      std::cout << "Matrix multiplication\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto b = ten::range<ten::matrix<float>>({3, 3});
      auto c = (a * b).eval();
      std::cout << c << std::endl;
   }

   {
      std::cout << "Matrix vector multiplication\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto b = ten::range<ten::vector<float>>({3});
      auto c = (a * b).eval();
      std::cout << c << std::endl;
   }

   {
      std::cout << "Scalar tensor multiplication\n";
      auto a = ten::range<ten::matrix<float>>({3, 3});
      auto b = 2.0f * a;
      std::cout << b.eval() << std::endl;
   }

   {
      std::cout << "Chaining\n";
      auto x = ten::range<ten::matrix<float>>({3, 3});
      auto y = ten::range<ten::matrix<float>>({3, 3});
      // UnaryExpr * matrix
      auto a = ten::sqrt(x);
      auto b = a * y;
      std::cout << b.eval() << std::endl;
      // matrix * UnaryExpr
      auto c = y * a;
      std::cout << c.eval() << std::endl;
      // Unarexpr * UnaryExpr
      auto d = a * a;
      std::cout << d.eval() << std::endl;
      // BinaryExpr * matrix
      auto e = x * x;
      auto f = e * x;
      std::cout << e.eval() << std::endl;
      std::cout << (x * x * x).eval() << std::endl;
      // matrix * BinaryExpr
      auto g = x * e;
      std::cout << g.eval() << std::endl;
   }

   {
      std::cout << "Chaining vector\n";
      auto x = ten::range<ten::matrix<float>>({3, 3});
      auto y = ten::range<ten::vector<float>>({3});
      // UnaryExpr * vector
      auto a = ten::sqrt(x);
      auto b = a * y;
      std::cout << b.eval() << std::endl;
      // BinaryExpr * vector
      std::cout << (x * x * y).eval() << std::endl;
   }

   {
      std::cout << "Functions that returns a scalar\n";
      auto x = ten::range<ten::vector<float>>({10});
      auto y = ten::min(x);
      std::cout << y.eval() << std::endl;
      auto z = ten::max(x);
      std::cout << z.eval() << std::endl;
      auto a = ten::prod(x);
      std::cout << a.eval() << std::endl;
      auto b = ten::cum_sum(x);
      std::cout << b.eval() << std::endl;
   }

}
