#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <Ten/Distributions.hxx>
#include <Ten/Tensor.hxx>
#include <Ten/Tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Uniform, FloatUniform) {
   ten::Uniform unif;

   float x = sample(unif);
   std::cout << "sampled from unif " << x << std::endl;
   size_t n = 10;
   Vector<float> y = sample(unif, n);
   for (size_t i = 0; i < n; i++) {
      std::cout << y[i] << " ";
   }
   std::cout << std::endl;
}

TEST(Normal, StandardNormal) {
   ten::Normal norm;

   float x = sample(norm);
   std::cout << "sampled from normal " << x << std::endl;
   size_t n = 10;
   Vector<float> y = sample(norm, n);
   for (size_t i = 0; i < n; i++) {
      std::cout << y[i] << " ";
   }
   std::cout << std::endl;
}

#endif
