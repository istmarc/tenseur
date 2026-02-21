#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <nanobench.h>

int main(int argc, char **argv) {
   if (argc > 2) {
      std::cerr << "./TenseurBench [file_name]" << std::endl;
      return 1;
   }
   std::string file_name = (argc == 2) ? std::string(argv[1]) : "torchBench";

   ankerl::nanobench::Bench bench;
   bench.title("Torch");

   std::vector<long> sizes = {64, 128, 256, 512, 1024, 2048, 4096};


   for (auto N : sizes) {
      std::cout << "Benchmarks for size = " << N << std::endl;
      auto a = torch::arange(N*N).reshape({N, N}).to(torch::kFloat32);
      auto b = torch::arange(N*N).reshape({N, N}).to(torch::kFloat32);
      auto c = torch::arange(N*N).reshape({N, N}).to(torch::kFloat32);
      bench.run("Gemm", [&] { c = torch::matmul(a, b); });
      bench.run("Gemm2", [&] {
         torch::Tensor d = torch::matmul(a, b);
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // sum
      bench.run("Sum", [&] { c = a + b; });
      bench.run("Sum2", [&] {
         torch::Tensor d = a + b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // sub
      bench.run("Sub", [&] { c = a - b; });
      bench.run("Sub2", [&] {
         torch::Tensor d = a - b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });

      // div
      bench.run("Div", [&] { c = a / b; });
      bench.run("Div2", [&] {
         torch::Tensor d = a / b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   // mul
   for (auto N : sizes) {
      auto a = torch::arange(N*N).to(torch::kFloat32);
      auto b = torch::arange(N*N).to(torch::kFloat32);
      auto c = torch::arange(N*N).to(torch::kFloat32);
      bench.run("Mul", [&] {
         c = a * b;
      });
      bench.run("Mul2", [&] {
         torch::Tensor d = a * b;
         ankerl::nanobench::doNotOptimizeAway(d);
      });
   }

   std::ofstream file(file_name + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   return 0;
}
