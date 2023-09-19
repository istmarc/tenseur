#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <Ten/Tensor.hxx>
#include <Ten/Tests.hxx>

#include "Ref.hxx"

using namespace ten;

TEST(Add, Vector_Vector) {
   size_t size = 10;
   Vector<float> a = iota<float>({size});
   Vector<float> b = iota<float>({size});
   auto c_ref = ten::tests::add(a, b);
   auto c = (a + b).eval();

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Sub, Vector_Vector) {
   size_t size = 10;
   Vector<float> a = iota<float>({size});
   Vector<float> b(size);
   for (size_t i = 0; i < size; i++) {
      b[i] = 2 * i;
   }
   auto c_ref = ten::tests::sub(a, b);
   Vector<float> c = a - b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Mul, Vector_Vector) {
   size_t size = 10;
   Vector<float> a = iota<float>({size});
   Vector<float> b = iota<float>({size});
   auto c_ref = ten::tests::mul(a, b);
   Vector<float> c = a * b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Mul, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   auto a = iota<SVector<float, size>>();
   auto b = iota<SVector<float, size>>();
   auto c_ref = ten::tests::muls(a, b);
   SVector<float, size> c = a * b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Div, Vector_Vector) {
   size_t size = 10;
   Vector<float> a = iota<float>({size});
   Vector<float> b(size);
   for (size_t i = 0; i < size; i++) {
      b[i] = i + 1;
   }
   auto c_ref = ten::tests::div(a, b);
   Vector<float> c = a / b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

TEST(Div, StaticVector_StaticVector) {
   constexpr size_t size = 10;
   SVector<float, size> a = iota<SVector<float, size>>();
   SVector<float, size> b;
   for (size_t i = 0; i < size; i++) {
      b[i] = i + 1;
   }
   auto c_ref = ten::tests::divs(a, b);
   SVector<float, size> c = a / b;

   ASSERT_TRUE(tests::equal(c, c_ref));
}

#endif
