#ifndef TENSEUR_TESTS_BASIC_FUNCTIONS
#define TENSEUR_TESTS_BASIC_FUNCTIONS

#include <Ten/Tensor.hxx>
#include <Ten/Tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Flatten, Tensor) {
   auto a = iota<Tensor<float, 3>>({2, 3, 4});
   Vector<float> b = flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Reshape, StaticTensor) {
   auto a = iota<STensor<float, 2, 3, 4>>();
   STensor<float, 3, 8> b = reshape<Shape<3, 8>>(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Flatten, StaticTensor) {
   auto a = iota<STensor<float, 2, 3, 4>>();
   STensor<float, 24> b = flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}


#endif
