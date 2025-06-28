#include <ten/tensor.hxx>

int main() {
   using namespace ten;

   auto a = range<vector<float>>({10});
   auto b = range<float>({10});
   std::cout << a << std::endl;
   //float c = dot(a, b);
   //std::cout << c << std::endl;
   vector<float> c = (a * b).eval();
   std::cout << c << std::endl;
   auto d = range<float, 2>({2, 3});
   std::cout << d << std::endl;

   diagonal<float> f(3, 3);
   for (size_t i = 0; i < 3; i++) {
      f[i] = i + 1;
   }
   std::cout << f << std::endl;

}
