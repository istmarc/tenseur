#ifndef TENSEUR_TESTS_TENSOR_RANDOM
#define TENSEUR_TESTS_TENSOR_RANDOM

#include "ten/distributions.hxx"
#include <ten/tensor>
#include <ten/tests.hxx>
#include <ten/random.hxx>

TEST(Random, RandomFloatTensor) {
   using namespace ten;

   size_t seed = 1234;
   ten::set_seed(seed);
   auto x = ten::rand_norm<ten::smatrix<float, 2, 3>>();
   ten::set_seed(seed);
   auto y = ten::rand_norm<ten::matrix<float>>({2, 3});
   ten::set_seed(seed);
   auto z = ten::rand_norm<ten::tensor<float, 2>>({2, 3});

   ASSERT_TRUE(tests::same_values(x, y));

   ASSERT_TRUE(tests::same_values(x, z));
}

#endif
