#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <Ten/Distributions.hxx>
#include <Ten/Tensor.hxx>
#include <Ten/Tests.hxx>

using namespace ten;

TEST(SaveToCsvFile, Vector) {
   size_t n = 10;
   Vector<float> x = iota<float>({n});
   x.save("vector.csv");
}

TEST(SaveToCsvFile, Matrix) {
   size_t n = 10;
   size_t m = 3;
   Matrix<float> x = iota<Matrix<float>>({n, m});
   x.save("matrix.csv");
}

#endif
