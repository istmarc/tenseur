#include <fstream>
#include <iostream>
#include <nanobench.h>

#include <ten/tensor.hxx>

int main(int argc, char **argv) {
   if (argc > 2) {
      std::cerr << "./TenseurBench [file_name]" << std::endl;
      return 1;
   }
   std::string file_name = (argc == 2) ? std::string(argv[1]) : "tenseurBench";

   using namespace ten;

   ankerl::nanobench::Bench bench;
   bench.title("Tenseur");

   std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

   for (auto N : sizes) {
      auto a = range<matrix<float>>({N, N});
      auto b = range<matrix<float>>({N, N});
      auto c = zeros<matrix<float>>({N, N});
      bench.run("Gemm", [&] { c = a * b; });
      bench.run("Gemm2", [&] {
         matrix<float> d = a * b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // sum
      bench.run("Sum", [&] { c = a + b; });
      bench.run("Sum2", [&] {
         matrix<float> d = a + b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // sub
      bench.run("Sub", [&] { c = a - b; });
      bench.run("Sub2", [&] {
         matrix<float> d = a - b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // div
      bench.run("Div", [&] { c = a / b; });
      bench.run("Div2", [&] {
         matrix<float> d = a / b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // mul
   for (auto N : sizes) {
      auto a = range<vector<float>>({N * N});
      auto b = range<vector<float>>({N * N});
      auto c = zeros<vector<float>>({N * N});
      bench.run("Mul", [&] { c = a * b; });
      bench.run("Mul2", [&] {
         vector<float> d = a * b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   std::ofstream file(file_name + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   return 0;
}
