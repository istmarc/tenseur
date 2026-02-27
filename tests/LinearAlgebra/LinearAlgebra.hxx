#ifndef TENSEUR_TESTS_TENSOR_RANDOM
#define TENSEUR_TESTS_TENSOR_RANDOM

#include <ten/tensor>
#include <ten/tests.hxx>
#include <ten/linalg>

TEST(LinearAlgebra, QRFactorization) {
   using namespace ten;

   auto a = ten::range<matrix<double>>({4, 4});
   auto [q, r] = ten::linalg::qr(a);

   matrix<double> x = q * r;

   ASSERT_TRUE(tests::equal(a, x, 1e-3));
}

// FIXME LU factorization fails with double data type
TEST(LinearAlgebra, LUFactorization) {
   using namespace ten;

   auto a = ten::range<matrix<float>>({4, 4});
   auto [p, l, u] = ten::linalg::lu(a);

   matrix<float> x = p * l * u;

   ASSERT_TRUE(tests::equal(a, x, 1e-3));
}

TEST(LinearAlgebra, CholeskyFactorization) {
   using namespace ten;

   matrix<double> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});

   auto [l, u] = ten::linalg::cholesky(a);

   matrix<double> x = l * u;

   ASSERT_TRUE(tests::equal(a, x, 1e-3));
}

TEST(LinearAlgebra, SVDFactorization) {
   using namespace ten;

   matrix<double> a({3, 3},
      {4., 12., -16., 12., 37., -43., -16., -43., 98.});

   auto [u, s, vt] = ten::linalg::svd(a);

   auto sigma = ten::zeros<ten::matrix<double>>({3, 3});
   for (size_t i = 0; i < 3; i++) {
      sigma(i,i) = s[i];
   }

   matrix<double> x = u * sigma * vt;

   ASSERT_TRUE(tests::equal(a, x, 1e-3));
}


#endif
