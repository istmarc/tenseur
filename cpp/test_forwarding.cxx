#include <ten/tensor.hxx>
#include <ten/distributions.hxx>

int main() {
   using namespace ten;

   set_seed(1234);
   uniform unif;
   auto x = range<vector<float>>({5});
   auto y = x + unif.sample(5);
   auto z = y.eval();
   std::cout << z << std::endl;
}
