#ifndef TENSEUR_TESTS_BINARY_OPS_ADD
#define TENSEUR_TESTS_BINARY_OPS_ADD

#include <ten/distributions.hxx>
#include <ten/tensor.hxx>
#include <ten/tests.hxx>

using namespace ten;

TEST(SaveToMtxFile, Vector) {
   size_t n = 10;
   ten::vector<float> x = range<ten::vector<float>>({n});
   ten::save(x, "vector.mtx");
}

TEST(SaveToMtxFile, Matrix) {
   size_t n = 10;
   size_t m = 3;
   matrix<float> x = range<matrix<float>>({n, m});
   ten::save(x, "matrix.mtx");
}

#endif
