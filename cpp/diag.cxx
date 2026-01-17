#include <ten/tensor>
#include <ten/io>

int main() {
   ten::diagonal<float> a({2, 2});
   for (size_t i = 0; i < 2; i++) {
      a[i] = i + 1.;
   }
   auto b = ten::dense(a);
   std::cout << b << std::endl;

   auto z = (b * b).eval();
   std::cout << "z = " << z << std::endl;

}
