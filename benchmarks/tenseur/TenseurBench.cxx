#include <fstream>
#include <iostream>
#include <nanobench.h>

#include <Ten/Tensor>

int main(int argc, char **argv) {
   if (argc > 2) {
      std::cerr << "./TenseurBench [file_name]" << std::endl;
      return 1;
   }
   std::string fileName = (argc == 2) ? std::string(argv[1]) : "tenseurBench";

   using namespace ten;

   ankerl::nanobench::Bench bench;
   bench.title("Tenseur");

   std::vector<size_t> sizes = {512, 1024, 2048};

   for (auto N : sizes) {
      auto a = iota<Matrix<float>>({N, N});
      auto b = iota<Matrix<float>>({N, N});
      auto c = zeros<Matrix<float>>({N, N});
      bench.run("Gemm", [&] { c = a * b; });
      bench.run("Gemm2", [&] {
         Matrix<float> d = a * b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // sum
   for (auto N : sizes) {
      auto a = iota<Matrix<float>>({N, N});
      auto b = iota<Matrix<float>>({N, N});
      auto c = zeros<Matrix<float>>({N, N});
      bench.run("Sum", [&] { c = a + b; });
      bench.run("Sum2", [&] {
         Matrix<float> d = a + b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // sub
   for (auto N : sizes) {
      auto a = iota<Matrix<float>>({N, N});
      auto b = iota<Matrix<float>>({N, N});
      auto c = zeros<Matrix<float>>({N, N});
      bench.run("Sub", [&] { c = a - b; });
      bench.run("Sub2", [&] {
         Matrix<float> d = a - b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // div
   for (auto N : sizes) {
      auto a = iota<Matrix<float>>({N, N});
      auto b = iota<Matrix<float>>({N, N});
      auto c = zeros<Matrix<float>>({N, N});
      bench.run("Div", [&] { c = a / b; });
      bench.run("Div2", [&] {
         Matrix<float> d = a / b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // mul
   for (auto N : sizes) {
      auto a = iota<Vector<float>>({N * N});
      auto b = iota<Vector<float>>({N * N});
      auto c = zeros<Vector<float>>({N * N});
      bench.run("Mul", [&] { c = a * b; });
      bench.run("Mul2", [&] {
         Vector<float> d = a * b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   std::ofstream file(fileName + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   return 0;
}
