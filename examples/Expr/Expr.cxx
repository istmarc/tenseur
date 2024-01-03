#include "Ten/Storage/DenseStorage.hxx"
#include <ios>
#include <iostream>

#include <Ten/Expr.hxx>
#include <Ten/Functional.hxx>
#include <Ten/Shape.hxx>
#include <Ten/Tensor.hxx>
#include <memory>
#include <type_traits>

template <class T> void printTensor(const T &val) {
   std::cout << "[";
   for (size_t i = 0; i < val.size(); i++)
      std::cout << val[i] << " ";
   std::cout << "]\n";
}

int main() {
   using namespace ten;
   using namespace std;

   {
      cout << "UnaryExpr" << endl;
      Vector<float> x({3});
      for (size_t i = 0; i < 3; i++)
         x[i] = -float(i);

      auto e = abs(x).eval();

      cout << "abs(x) = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << e[i] << " ";
      cout << "]" << endl;

      cout << "And x = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << x[i] << " ";
      cout << "]" << endl;
   }

   {
      cout << "UnaryExpr min" << endl;
      Vector<float> x({3});
      for (size_t i = 0; i < 3; i++)
         x[i] = -float(i + 1.);
      auto e = min(x);
      auto v = e.eval();

      cout << "min(x) = [ ";
      float m = x[0];
      for (size_t i = 1; i < 3; i++) {
         if (x[i] < m)
            m = x[i];
      }
      cout << m << " ";
      cout << "]" << endl;

      cout << "And expr value = " << v.value() << endl;
   }

   {
      cout << "UnaryExpr sqrt" << endl;
      auto x = iota<Vector<float>>(3);
      for (size_t i = 0; i < 3; i++)
         x[i] = float(i);
      auto e = sqrt(x);
      auto out = e.eval();

      cout << "sqrt(x) = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << out[i] << " ";
      cout << "]" << endl;

      cout << "And x = [ ";
      for (size_t i = 0; i < 3; i++)
         cout << x[i] << " ";
      cout << "]" << endl;
   }

   {
      cout << "Binary expr a + b" << std::endl;
      auto a = iota<Vector<float>>(3);
      auto b = iota<Vector<float>>(3);
      auto c = a + b;
      auto res = c.eval();
      printTensor(a);
      printTensor(b);
      printTensor(res);
   }

   {
      cout << "Binary expr a - b" << std::endl;
      Vector<float> a({3});
      Vector<float> b({3});
      for (size_t i = 0; i < 3; i++) {
         a[i] = i;
         b[i] = i + 1;
      }
      auto c = (a - b).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), Tensor<float, 1>>);
   }

   {
      cout << "Binary expr a * b" << std::endl;
      auto a = iota<Vector<float>>(6);
      auto b = iota<Vector<float>>(6);
      auto c = (a * b).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), Tensor<float, 1>>);
   }

   {
      cout << "Binary expr A * B" << std::endl;
      auto a = iota<Matrix<float>>({2, 3});
      auto b = iota<Matrix<float>>({3, 4});
      auto c = (a * b).eval();
      cout << "A = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            std::cout << a(i, j) << " ";
         }
         cout << endl;
      }
      cout << "B = \n";
      for (size_t i = 0; i < 3; i++) {
         for (size_t j = 0; j < 4; j++) {
            std::cout << b(i, j) << " ";
         }
         cout << endl;
      }
      cout << "C = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 4; j++) {
            std::cout << c(i, j) << " ";
         }
         cout << endl;
      }
      static_assert(std::is_same_v<decltype(c), Tensor<float, 2>>);
   }

/*
   {
      cout << "Binary expr matrix * vector" << std::endl;
      auto a = iota<Matrix<float>>({2, 3});
      auto b = iota<Vector<float>>(3);
      auto c = (a * b).eval();
      cout << "A = \n";
      for (size_t i = 0; i < 2; i++) {
         for (size_t j = 0; j < 3; j++) {
            std::cout << a(i, j) << " ";
         }
         cout << endl;
      }
      cout << "B = \n";
      for (size_t j = 0; j < 3; j++) {
         std::cout << b[j] << " ";
      }
      cout << endl;
      cout << "C = \n";
      for (size_t j = 0; j < 2; j++) {
         std::cout << c[j] << " ";
      }
      cout << endl;
      static_assert(std::is_same_v<decltype(c), Tensor<float, 1>>);
   }
*/

   {
      cout << "Binary expr alpha * a" << std::endl;
      auto a = iota<Vector<float>>(5);
      auto c = (2. * a).eval();
      printTensor(c);
      static_assert(std::is_same_v<decltype(c), Tensor<float, 1>>);
   }

/*
   {
      cout << "Chain binary expressions" << std::endl;
      auto a = iota<Vector<float>>(5);
      auto b = iota<Vector<float>>(5);
      auto c = a + b;
      // 0 2 4  6  8
      // 0 2 8 18 32
      auto d = (c * a).eval();
      printTensor(d);
   }
*/

   {
      cout << "Chain unary expressions" << std::endl;
      auto a = iota<Vector<float>>(5);
      auto b = sqrt(a);
      auto c = sqrt(b).eval();
      printTensor(c);
   }

   {
      cout << "Chain unary and binary expressions" << std::endl;
      auto a = iota<Vector<float>>(5);
      auto b = 2. * a;
      auto c = sqrt(b);
      auto d = (c + a).eval();
      printTensor(d);
   }

   {
      cout << "Tensor from unary expr" << std::endl;
      auto a = iota<Vector<float>>(5);
      Vector<float> b = sqrt(a);
      printTensor(b);
   }

   {
      cout << "Tensor from binary expr" << std::endl;
      auto a = iota<StaticVector<float, 5>>();
      auto b = iota<StaticVector<float, 5>>();
      StaticVector<float, 5> d = a + b;
      printTensor(d);
   }

   return 0;
}
