#ifndef TENSEUR_TESTS_TENSOR_TRAITS
#define TENSEUR_TESTS_TENSOR_TRAITS

#include <ten/tensor>

struct EmptyStruct {};

template <typename> struct TemplatedStruct {};

template <typename...> struct VariadicStruct {};

TEST(Traits, isTensor) {
   using namespace ten;

   using Tensor_t = ten::tensor<float, 3>;
   ASSERT_TRUE(is_tensor<Tensor_t>::value);

   ASSERT_TRUE(!is_tensor<int>::value);

   ASSERT_TRUE(!is_tensor<EmptyStruct>::value);

   ASSERT_TRUE(!is_tensor<TemplatedStruct<size_t>>::value);

   using VariadicStruct_t = VariadicStruct<float, size_t>;
   ASSERT_TRUE(!is_tensor<VariadicStruct_t>::value);
}

#endif
