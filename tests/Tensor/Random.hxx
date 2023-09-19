#ifndef TENSEUR_TESTS_TENSOR_RANDOM
#define TENSEUR_TESTS_TENSOR_RANDOM

#include <Ten/Tensor>
#include <Ten/Tests.hxx>

TEST(Random, RandomFloatTensor) {
   using namespace ten;

   size_t seed = 1234;
   auto x = rand<StaticMatrix<float, 2, 3>>(seed);
   auto y = rand<Matrix<float>>({2, 3}, seed);
   auto z = rand<float, 3>({2, 3, 4}, seed);

   ASSERT_TRUE(tests::same_values(x, y));

   ASSERT_TRUE(tests::same_values(x, z));
}

#endif
