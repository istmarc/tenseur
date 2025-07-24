#include <ten/linalg>
#include <ten/tensor>

int main() {
   auto A = ten::matrix<float>(
       {3, 3}, {3.0f, 2.0f, -1.0f, 2.0f, -2.0f, 0.5f, -1.f, 4.f, -1.f});
   auto y = ten::vector<float>({3}, {1.f, -2.f, 0.f});
   using namespace ten::linalg;

   {
      ls_options options(ls_method::qr);
      ten::linalg::linear_system<float> ls(std::move(options));
      ls.solve(A, y);
      auto x = ls.solution();

      // must be [1, -2, -2]
      std::cout << "Solution using QR:\n";
      std::cout << x << std::endl;
   }

   {
      ls_options options(ls_method::lu);
      ten::linalg::linear_system<float> ls(std::move(options));
      ls.solve(A, y);
      auto x = ls.solution();
      std::cout << "Solution using LU:\n";
      std::cout << x << std::endl;
   }

   {
      ls_options options(ls_method::svd);
      ten::linalg::linear_system<float> ls(std::move(options));
      ls.solve(A, y);
      auto x = ls.solution();
      std::cout << "Solution using SVD:\n";
      std::cout << x << std::endl;
   }
}
