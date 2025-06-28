#include <Ten/Types.hxx>
#include <Ten/Tensor.hxx>
#include <Ten/Distributions.hxx>
#include <Ten/Statistics/Statistics.hxx>

int main() {
   using namespace ten;
   ten::Normal<> norm;
   setSeed(1234);

   Vector<float> x = norm.sample(1000);
   std::cout << mean(x) << std::endl;
   std::cout << ten::std(x) << std::endl;
}
