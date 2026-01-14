#include <ten/tensor>
#include <ten/io>

int main() {
   using namespace ten;

   auto a = range<vector<float>>({100});
   auto b = cum_sum(a).eval();
   std::cout << a << std::endl;
   std::cout << b << std::endl;
}

