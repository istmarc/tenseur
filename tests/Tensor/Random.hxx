#ifndef TENSEUR_TESTS_TENSOR_RANDOM
#define TENSEUR_TESTS_TENSOR_RANDOM

#include <ten/tensor>
#include <ten/tests.hxx>
#include <ten/random.hxx>

/// FIXME This test fails
TEST(Random, RandomFloatTensor) {
   using namespace ten;

   size_t seed = 1234;
   auto x = ten::rand<ten::smatrix<float, 2, 3>>(seed);
   auto y = ten::rand<ten::matrix<float>>({2, 3}, seed);
   auto z = ten::rand<ten::tensor<float, 3>>({2, 3, 4}, seed);

   ASSERT_TRUE(tests::same_values(x, y));

   ASSERT_TRUE(tests::same_values(x, z));
}

#endif
