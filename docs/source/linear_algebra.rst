Linear Algebra
==============

Factorization routines
----------------------

- QR factorization

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>
   #include <ten/linalg>

   int main() {
      auto a = ten::range<ten::matrix<float>>({4, 4});

      auto [q, r] = ten::linalg::qr(a);

      std::cout << q << std::endl;
      std::cout << r << std::endl;

      std::cout << (q * r).eval() << std::endl;
      std::cout << a << std::endl;
   }

- LU factorization

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>
   #include <ten/linalg>

   int main() {
      auto a = ten::range<ten::matrix<float>>({4, 4});

      auto [p, l, u] = ten::linalg::lu(a);

      std::cout << "PA = LU" << std::endl;
      ten::matrix<float> x = p * l * u;
      std::cout << x << std::endl;
      ::ten::matrix<float> v = a - x;
      std::cout << ten::all_close(v) << std::endl;
   }

- Cholesky factorization

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>
   #include <ten/linalg>

   int main() {

      ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
         -16., -43., 98.});

      auto [l, u] = ten::linalg::cholesky(a);

      std::cout << "LU" << std::endl;
      ten::matrix<float> lu = l * u;
      std::cout << lu << std::endl;

      ::ten::matrix<float> v = (l*u).eval() - a;
      std::cout << ten::all_close(v) << std::endl;
   }

- SVD decomposition

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/linalg>

   int main() {

      ten::matrix<float> a({3, 3}, {4., 12., -16., 12., 37., -43.,
         -16., -43., 98.});

      auto [u, s, vt] = ten::linalg::svd(a);

      std::cout << u << std::endl;
      std::cout << s << std::endl;
      std::cout << vt << std::endl;
   }

