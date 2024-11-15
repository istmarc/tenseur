#ifndef TENSEUR_TESTS_BASIC_FUNCTIONS
#define TENSEUR_TESTS_BASIC_FUNCTIONS

#include <ten/tensor.hxx>
#include <ten/tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Flatten, Tensor) {
   auto a = ten::range<ten::tensor<float, 3>>({2, 3, 4});
   ten::vector<float> b = ten::flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Reshape, StaticTensor) {
   auto a = range<stensor<float, 2, 3, 4>>();
   ten::stensor<float, 3, 8> b = reshape<ten::shape<3, 8>>(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Flatten, StaticTensor) {
   auto a = range<stensor<float, 2, 3, 4>>();
   stensor<float, 24> b = flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}


#endif
