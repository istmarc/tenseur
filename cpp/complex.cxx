#include <ten/tensor.hxx>
#include <complex>

int main(){
   using namespace ten;

   matrix<std::complex<float>> a({2, 2});
   std::cout << a << std::endl;
   //std::cout << a(0,0) << std::endl;

}
