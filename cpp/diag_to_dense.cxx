#include <ten/tensor>

int main() {
   ten::sdiagonal<float, 3, 3> x;
   for (size_t i = 0; i < 3; i++) {
      x[i] = i + 1.;
   }
   std::cout << x << std::endl;

   auto y = ten::dense(x);
   std::cout << y << std::endl;

   ten::diagonal<float> a({2, 2});
   for (size_t i = 0; i < 2; i++) {
      a[i] = i + 1.;
   }
   auto b = ten::dense(a);
   std::cout << b << std::endl;

   auto z = (b * b).eval();
   std::cout << "z = " << z << std::endl;

}
