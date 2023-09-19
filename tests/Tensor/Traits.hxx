#ifndef TENSEUR_TESTS_TENSOR_TRAITS
#define TENSEUR_TESTS_TENSOR_TRAITS

#include <Ten/Tensor>

struct EmptyStruct {};

template <typename> struct TemplatedStruct {};

template <typename...> struct VariadicStruct {};

TEST(Traits, isTensor) {
   using namespace ten;

   using Tensor_t = Tensor<float, 3>;
   ASSERT_TRUE(isTensor<Tensor_t>::value);

   ASSERT_TRUE(!isTensor<int>::value);

   ASSERT_TRUE(!isTensor<EmptyStruct>::value);

   ASSERT_TRUE(!isTensor<TemplatedStruct<size_t>>::value);

   using VariadicStruct_t = VariadicStruct<float, size_t>;
   ASSERT_TRUE(!isTensor<VariadicStruct_t>::value);
}

#endif
