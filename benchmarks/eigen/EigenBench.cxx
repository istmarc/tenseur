#include <fstream>
#include <iostream>
#include <nanobench.h>

#include <Eigen/Core>
using namespace Eigen;

int main(int argc, char **argv) {
   if (argc > 2) {
      std::cerr << "./EigenBench [file_name]" << std::endl;
      return 1;
   }
   std::string file_name = (argc == 2) ? std::string(argv[1]) : "eigenBench";

   ankerl::nanobench::Bench bench;
   bench.title("Eigen");

   std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

   for (auto N : sizes) {
      MatrixXf a(N, N);
      MatrixXf b(N, N);
      MatrixXf c = MatrixXf::Zero(N, N);
      size_t k = 0;
      for (size_t i = 0; i < N; i++) {
         for (size_t j = 0; j < N; j++) {
            a(j, i) = k;
            b(j, i) = k;
            k++;
         }
      }

      bench.run("Gemm", [&] { c = a * b; });
      bench.run("Gemm2", [&] { MatrixXf d = a * b; });

      bench.run("Sum", [&] { c = a + b; });
      bench.run("Sum2", [&] { MatrixXf d = a + b; });

      bench.run("Sub", [&] { c = a - b; });
      bench.run("Sub2", [&] { MatrixXf d = a - b; });

      bench.run("Div", [&] { c = a.array() / b.array(); });
      bench.run("Div2", [&] { MatrixXf d = a.array() / b.array(); });

      bench.run("Mul", [&] { c = a.array() * b.array(); });
      bench.run("Mul2", [&] { MatrixXf d = a.array() * b.array(); });
   }

   std::ofstream file(file_name + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   return 0;
}
