#ifndef TENSEUR_TESTS_TENSOR_CAST
#define TENSEUR_TESTS_TENSOR_CAST

#include <ten/tensor.hxx>
#include <ten/tests.hxx>
#include <ten/types.hxx>

template <class From, class To, size_t... dims>
testing::AssertionResult testCastStaticTensor() {
   using tensor_type = ::ten::stensor<From, dims...>;
   tensor_type a = ::ten::range<tensor_type>();
   auto b = ::ten::cast<To>(a);
   // Types
   testing::StaticAssertTypeEq<decltype(b), ::ten::stensor<To, dims...>>();
   // Values
   return ::ten::tests::same_values(a, b);
}

template <typename> class Cast : public ::testing::Test {};
TYPED_TEST_SUITE_P(Cast);

template <class From, class To> struct TypesHolder {
   using from_type = From;
   using to_type = To;
};

TYPED_TEST_P(Cast, castStaticTensor) {
   using namespace ten;
   using from_type = typename TypeParam::from_type;
   using to_type = typename TypeParam::to_type;

   auto result_a = testCastStaticTensor<from_type, to_type, 2>();
   ASSERT_TRUE(result_a);
   auto result_b = testCastStaticTensor<from_type, to_type, 2, 3>();
   ASSERT_TRUE(result_b);
   auto result_c = testCastStaticTensor<from_type, to_type, 2, 3, 4>();
   ASSERT_TRUE(result_c);
   auto result_d = testCastStaticTensor<from_type, to_type, 2, 3, 4, 5>();
   ASSERT_TRUE(result_d);
}

// Register the test suite
REGISTER_TYPED_TEST_SUITE_P(Cast, castStaticTensor);
// Instantiate the pattern with the types
// TODO Pretty printing format
using types =
    ::testing::Types<TypesHolder<float, double>, TypesHolder<double, float>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Cast, Cast, types);

#endif
