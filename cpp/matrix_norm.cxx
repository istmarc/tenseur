#include <ten/tensor>
#include <ten/linalg>

int main() {
   using namespace ten;
   ten::matrix<float> m({3, 3}, {-3., 2., 0., 5., 6., 2., 7., 4., 8.});
   std::cout << m << std::endl;

   std::cout << "Norms" << std::endl;
   std::cout << ten::norm(m, ten::matrix_norm::frobenius) << std::endl;
   std::cout << ten::norm(m, ten::matrix_norm::l1) << std::endl;
   std::cout << ten::norm(m, ten::matrix_norm::linf) << std::endl;

}
