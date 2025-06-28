#include "Ten/Shape.hxx"
#include <Ten/Tensor.hxx>

int main() {
   using namespace ten;
   ten::DynamicShape<2> shape{2, 2};
   Matrix<float> a(shape, {1., 2., 3., 4.});
   std::cout << a << std::endl;

   Matrix<float> b({2ul, 2ul}, {0.f, 1.f, 2.f, 3.f});
   std::cout << b << std::endl;

   DiagonalMatrix<float> d({2, 2}, {1., 2.});
   std::cout << d << std::endl;

}
