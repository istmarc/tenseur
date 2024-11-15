#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <ten/distributions.hxx>
#include <ten/tensor.hxx>
#include <ten/tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Uniform, FloatUniform) {
   ten::uniform unif;

   float x = unif.sample();
   std::cout << "sampled from unif " << x << std::endl;
   size_t n = 10;
   ten::vector<float> y = unif.sample(n);
   for (size_t i = 0; i < n; i++) {
      std::cout << y[i] << " ";
   }
   std::cout << std::endl;
}

TEST(Normal, StandardNormal) {
   ten::Normal norm;

   float x = norm.sample();
   std::cout << "sampled from normal " << x << std::endl;
   size_t n = 10;
   ten::vector<float> y = norm.sample(n);
   for (size_t i = 0; i < n; i++) {
      std::cout << y[i] << " ";
   }
   std::cout << std::endl;
}

#endif
