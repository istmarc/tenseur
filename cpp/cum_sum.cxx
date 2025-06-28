#include <ten/tensor.hxx>

int main() {
   using namespace ten;

   auto a = range<vector<float>>({100});
   auto b = cum_sum(a).eval();
   std::cout << a << std::endl;
   std::cout << b << std::endl;

   auto d = (a - a).eval();
   std::cout << std::boolalpha << all_close(d, 1e-5) << std::endl;

}
