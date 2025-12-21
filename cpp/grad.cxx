#include <ios>
#include <ten/tensor>
#include <ten/io>

void f() {
   std::cout <<"=================================\n";
}

int main() {
   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      std::cout << x << std::endl;
      std::cout << x.grad() << std::endl;
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      std::cout << x << std::endl;
      auto y = ten::sqrt(x);
      y.backward();
      std::cout << "Gradient of sqrt\n";
      std::cout << x.grad() << std::endl;
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      std::cout << x << std::endl;
      auto y = ten::sqrt(x);
      auto z = ten::sqr(y);
      z.eval();
      std::cout << z.input().input() << std::endl;
      std::cout << z.input().value() << std::endl;

      std::cout << "z value =\n" << z.value() << std::endl;
      std::cout << std::boolalpha << z.evaluated() << std::endl;
      std::cout << std::boolalpha << y.evaluated() << std::endl;
      std::cout << "y value =\n" << y.value() << std::endl;
      z.backward();
      std::cout << "And the gradients\n";
      std::cout << x.grad() << std::endl;
      std::cout << y.grad() << std::endl;
      std::cout << z.grad() << std::endl;
   }

   /*

   {
      ten::stensor<float, 10> x({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
      std::cout << x << std::endl;
      auto y = ten::sqrt(x);
      y.backward();
      auto grad = x.grad();
      std::cout << grad << std::endl;
   }*/

}

