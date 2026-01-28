#include <ten/tensor>
#include <ten/linalg>
#include <ten/io>

int main() {
   ten::matrix<float> A(
       {3, 3}, {3.0f, 2.0f, -1.0f, 2.0f, -2.0f, 0.5f, -1.f, 4.f, -1.f});
   ten::vector<float> y({3}, {1.f, -2.f, 0.f});
   using namespace ten::linalg;

   {
      ten::linalg::linear_system<float> ls(ls_options{ls_method::qr});
      ls.solve(A, y);
      auto x = ls.solution();

      // must be [1, -2, -2]
      std::cout << "Solution using QR:\n";
      std::cout << x << std::endl;
   }

   {
      ten::linalg::linear_system<float> ls(ls_options{ls_method::lu});
      ls.solve(A, y);
      auto x = ls.solution();
      std::cout << "Solution using LU:\n";
      std::cout << x << std::endl;
   }

   {
      ls_options options(ls_method::svd);
      ten::linalg::linear_system<float> ls(options);
      ls.solve(A, y);
      auto x = ls.solution();
      std::cout << "Solution using SVD:\n";
      std::cout << x << std::endl;
   }
}
