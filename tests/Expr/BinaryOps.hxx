#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <ten/tensor.hxx>
#include <ten/tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Add, Vector_Vector) {
   size_t size = 10;
   ten::vector<float> a = range<ten::vector<float>>({size});
   ten::vector<float> b = range<ten::vector<float>>({size});
   auto c_ref = ten::tests::add(a, b);
   auto c = (a + b).eval();

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Add, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   auto a = range<ten::svector<float, size>>();
   auto b = range<ten::svector<float, size>>();
   auto c_ref = ten::tests::adds(a, b);
   ten::svector<float, size> c = a + b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Sub, Vector_Vector) {
   size_t size = 10;
   ten::vector<float> a = range<ten::vector<float>>({size});
   ten::vector<float> b(size);
   for (size_t i = 0; i < size; i++) {
      b[i] = 2 * i;
   }
   auto c_ref = ten::tests::sub(a, b);
   ten::vector<float> c = a - b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Sub, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   auto a = range<ten::svector<float, size>>();
   auto b = range<ten::svector<float, size>>();
   auto c_ref = ten::tests::subs(a, b);
   ten::svector<float, size> c = a - b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}


TEST(Mul, Vector_Vector) {
   size_t size = 10;
   ten::vector<float> a = range<ten::vector<float>>({size});
   ten::vector<float> b = range<ten::vector<float>>({size});
   auto c_ref = ten::tests::mul(a, b);
   ten::vector<float> c = a * b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Mul, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   auto a = range<ten::svector<float, size>>();
   auto b = range<ten::svector<float, size>>();
   auto c_ref = ten::tests::muls(a, b);
   ten::svector<float, size> c = a * b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Div, Vector_Vector) {
   size_t size = 10;
   ten::vector<float> a = range<ten::vector<float>>({size});
   ten::vector<float> b(size);
   for (size_t i = 0; i < size; i++) {
      b[i] = i + 1;
   }
   auto c_ref = ten::tests::div(a, b);
   ten::vector<float> c = a / b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Div, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   ten::svector<float, size> a = range<ten::svector<float, size>>();
   ten::svector<float, size> b;
   for (size_t i = 0; i < size; i++) {
      b[i] = i + 1;
   }
   auto c_ref = ten::tests::divs(a, b);
   ten::svector<float, size> c = a / b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

#endif
