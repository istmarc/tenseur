#include <ten/tensor>
#include <ten/io>

void f() {
   std::cout <<"=================================\n";
}

int main() {
   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      std::cout << x << std::endl;
      std::cout << x.grad() << std::endl;
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      std::cout << x << std::endl;
      auto y = ten::sqrt(x);
      y.backward(true);
      std::cout << "Gradient of sqrt\n";
      std::cout << x.grad() << std::endl;
   }

   /*
   {
      f();
      ten::scalar<float> x(2.0f);
      auto y = ten::sqrt(x);
      auto z = ten::sqr(y);
      z.backward();
   }*/

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      std::cout << x << std::endl;
      auto y = ten::sqr(x);
      auto z = ten::sin(y);
      auto t = ten::cos(z);
      t.eval();
      t.backward(true);
      std::cout << "And the gradients\n";
      std::cout << x.grad() << std::endl;
      // Should be equal to
      //-0.805725
      //-1.79517
      //2.18973
      //-2.17536
      //1.30805
      std::cout << y.grad() << std::endl;
      std::cout << z.grad() << std::endl;
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      std::cout << x << std::endl;
      auto y = ten::sqr(x);
      auto z = ten::sin(y);
      auto t = ten::cos(z);
      t.eval();
      t.backward();
      std::cout << "And the gradients\n";
      std::cout << x.grad() << std::endl;
      // Should be equal to
      //-0.805725
      //-1.79517
      //2.18973
      //-2.17536
      //1.30805
      std::cout << y.has_retain_grad() << std::endl;
      std::cout << z.has_retain_grad() << std::endl;
      std::cout << t.has_retain_grad() << std::endl;
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      std::cout << x << std::endl;
      auto y = ten::sqr(x);
      auto z = ten::sin(y);
      z.retain_grad();
      auto t = ten::cos(z);
      t.eval();
      t.backward();
      std::cout << "And the gradients\n";
      std::cout << x.grad() << std::endl;
      // Should be equal to
      //-0.805725
      //-1.79517
      //2.18973
      //-2.17536
      //1.30805
      std::cout << z.has_retain_grad() << std::endl;
      std::cout << y.grad() << std::endl;
      // [-0.402862, ..., 0.130805]
   }

   // Scalar functions
   {
      f();
      ten::scalar<float> x(2.0f, true);
      auto y = ten::sin(x);
      auto z = ten::cos(y);
      z.eval();
      std::cout << "z value = " << z.value() << std::endl;
      // 0.6143
      z.backward(true);
      std::cout << x.grad() << std::endl;
      // 0.32837
   }

   // Scalar functions with create_graph=false
   {
      f();
      ten::scalar<float> x(2.0f, true);
      auto y = ten::sin(x);
      auto z = ten::cos(y);
      z.eval();
      std::cout << "z value = " << z.value() << std::endl;
      // 0.6143
      z.backward();
      std::cout << x.grad() << std::endl;
      // 0.32837
   }


   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      ten::vector<float> y({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      auto z = x / y;
      auto t = ten::sum(z);
      t.eval();
      t.backward(true);
      std::cout << "The gradients\n";
      std::cout << x.grad() << std::endl;
      // 1, 0.5, 0.333333, 0.25, 0.2
      std::cout << y.grad() << std::endl;
      // -1, -0.5, -0.333333, -0.25, -0.2
   }

   {
      f();
      ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      ten::vector<float> y({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
      auto z = x / y;
      auto t = ten::sum(z);
      t.eval();
      t.backward();
      std::cout << "The gradients\n";
      std::cout << x.grad() << std::endl;
      // 1, 0.5, 0.333333, 0.25, 0.2
      std::cout << y.grad() << std::endl;
      // -1, -0.5, -0.333333, -0.25, -0.2
   }

   {
      f();
      ten::stensor<float, 10> x({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
      std::cout << x << std::endl;
      auto y = ten::sqrt(x);
      y.eval();
      y.backward();
      std::cout << y.value() << std::endl;
      std::cout << x.grad() << std::endl;
   }


   /*
   {
      f();
      ten::matrix<float> x({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      ten::matrix<float> y({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      auto z = x * y;
      z.eval();
      std::cout << z.value() << std::endl;
   }
   */
}

