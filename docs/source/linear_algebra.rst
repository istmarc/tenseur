Linear Algebra
==============

Factorization routines
----------------------

- QR factorization

.. code-block:: cpp

   auto a = ten::range<ten::matrix<float>>({4, 4});

   ten::qr qr_fact;
   qr_fact.factorize(a);
   auto q = qr_fact.q();
   auto r = qr_fact.r();

   std::cout << q << std::endl;
   std::cout << r << std::endl;

   std::cout << (q * r).eval() << std::endl;
   std::cout << a << std::endl;

- LU factorization

.. code-block:: cpp

   auto a = ten::range<ten::matrix<float>>({4, 4});

   ten::lu lu_fact;
   lu_fact.factorize(a);

   auto p = lu_fact.p();
   auto l = lu_fact.l();
   auto u = lu_fact.u();

   std::cout << "PA = LU" << std::endl;
   ten::matrix<float> pa = p * a;
   std::cout << pa << std::endl;
   ten::matrix<float> lu = l * u;
   std::cout << lu << std::endl;
   ::ten::matrix<float> v = pa - lu;
   std::cout << ten::all_close(v) << std::endl;

- Cholesky factorization

.. code-block:: cpp

   ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});

   ten::cholesky cholesky_fact;
   cholesky_fact.factorize(a);

   auto l = cholesky_fact.l();
   auto u = cholesky_fact.u();

   std::cout << "LU" << std::endl;
   ten::matrix<float> lu = l * u;
   std::cout << lu << std::endl;

   ::ten::matrix<float> v = (l*u).eval() - a;
   std::cout << ten::all_close(v) << std::endl;

- SVD decomposition

.. code-block:: cpp

   ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
      -16., -43., 98.});

   ten::svd svd_fact;
   svd_fact.factorize(a);

   auto u = svd_fact.u();
   auto s = svd_fact.sigma();
   auto vt = svd_fact.vt();

   std::cout << u << std::endl;
   std::cout << s << std::endl;
   std::cout << vt << std::endl;

