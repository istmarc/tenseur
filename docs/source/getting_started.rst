Getting started
===============

API
---
Tenseur has an easy to use api, It can be illustrated with the following example:

.. code-block:: cpp

   // Normal distribution
   ten::normal norm;
   // Sample from a normal distribution
   ten::vector<float> x = norm.sample(1000);
   // Save to a mtx file (Matrix Market format)
   ten::save_mtx(x, "norm.mtx");

The saved file can be loaded in numpy:

.. code-block:: python

   import numpy as np
   import scipy as sp
   import matplotlib.pyplot as plt
   import seaborn as sn
   plt.style.use("science")
   a = sp.io.mmread("norm.mtx").flatten()
   plt.hist(a, color = "black")
   plt.show()

.. image:: _images/hist.png

Constructors
------------

The following constructors are defined for vectors, matrices, and tensors:

Vector
------

.. code-block:: cpp

   // Uninitialized vector
   ten::vector<T> x({size});
   // Vector initialized with data
   ten::vector<T> y({size}, data);

Static vector
-------------

.. code-block:: cpp

   ten::svector<float, size> x;
   ten::svector<float, size> y(data);

Matrix
------

.. code-block:: cpp

   ten::matrix<float> x(shape);
   ten::matrix<float> y(shape, data);

Static matrix
-------------

.. code-block:: cpp

   ten::smatrix<float, dims...> x;
   ten::smatrix<float, dims...> y(data);

Tensor
------

.. code-block:: cpp

   ten::tensor<float> x(shape);
   ten::tensor<float> y(shape, data);

Static tensor
-------------

.. code-block:: cpp

   ten::stensor<float, dims...> x;
   ten::stensor<float, dims...> y(data);

Special matrices
----------------

- Transposed

.. code-block:: cpp

   ten::matrix<float> x = ten::range<ten::matrix<float>>(shape);
   auto y = ten::transposed(x);
   std::cout << std::boolalpha << y.is_transposed() << std::endl;

- Symmetric

.. code-block:: cpp

   ten::matrix<float> x(shape, data);
   auto y = ten::symmetric(x);
   std::cout << std::boolalpha << y.is_symmetric() << std::endl;

- Hermitian

.. code-block:: cpp

   ten::matrix<std::complex<float>> x(shape, data);
   auto y = ten::hermitian(x);
   std::cout << std::boolalpha << y.is_hermitian() << std::endl;

- Lower triangular

.. code-block:: cpp

   ten::matrix<float> x(shape, data);
   auto y = ten::lower_tr(x);
   std::cout << std::boolalpha << y.is_lower_tr() << std::endl;

- Upper triangular

.. code-block:: cpp

   ten::matrix<float> x(shape, data);
   auto y = ten::upper_tr(x);
   std::cout << std::boolalpha << y.is_upper_tr() << std::endl;

