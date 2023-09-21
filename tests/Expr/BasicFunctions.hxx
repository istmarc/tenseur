#ifndef TENSEUR_TESTS_BASIC_FUNCTIONS
#define TENSEUR_TESTS_BASIC_FUNCTIONS

#include <Ten/Tensor.hxx>
#include <Ten/Tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Sqrt, StaticVector) {
   auto a = iota<SVector<float, 10>>();
   SVector<float, 10> b = sqrt(a);

   SVector<float, 10> ref;
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::sqrt(i);
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Sqrt, Vector) {
   auto a = iota<Vector<float>>(10);
   Vector<float> b = sqrt(a);

   Vector<float> ref(10);
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::sqrt(i);
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Abs, StaticVector) {
   auto a = iota<SVector<float, 10>>();
   SVector<float, 10> b = abs(a);

   SVector<float, 10> ref;
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::abs(float(i));
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Abs, Vector) {
   auto a = iota<Vector<float>>(10);
   Vector<float> b = abs(a);

   Vector<float> ref(10);
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::abs(float(i));
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Pow, StaticVector) {
   auto a = iota<SVector<float, 10>>();
   SVector<float, 10> b = pow(a, 3);

   SVector<float, 10> ref;
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::pow(i, 3);
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Pow, Vector) {
   auto a = iota<Vector<float>>(10);
   Vector<float> b = pow(a, 3);

   Vector<float> ref(10);
   for (std::size_t i = 0; i < 10; i++) {
      ref[i] = std::pow(i, 3);
   }

   ASSERT_TRUE(tests::equal(b, ref));
}

TEST(Flatten, Tensor) {
   auto a = iota<Tensor<float, 3>>({2, 3, 4});
   Vector<float> b = flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Flatten, StaticTensor) {
   auto a = iota<STensor<float, 2, 3, 4>>();
   STensor<float, 24> b = flatten(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

TEST(Reshape, StaticTensor) {
   auto a = iota<STensor<float, 2, 3, 4>>();
   STensor<float, 3, 8> b = reshape<Shape<3, 8>>(a);

   ASSERT_TRUE(tests::same_values(a, b));
}

#endif
