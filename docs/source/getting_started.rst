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
   save(x, "norm.mtx");

The saved file can be loaded in numpy:

.. code-block:: python

   import numpy as np
   import scipy as sp
   import matplotlib.pyplot as plt
   import seaborn as sn
   plt.style.use("science")
   a = sp.io.mmread("norm.mtx").flatten()
   plt.hist(a, color = "black")
   plt.savefig("hist.png")
   plt.show()

.. image:: _images/hist.png

Constructors
------------

The following constructors are defined for vectors, matrices, and tensors:

Vector
------

.. code-block:: cpp

   size_t size = 5;
   // Uninitialized vector
   vector<T> x({size});
   // Vector initialized with data
   vector<T> y({size}, {0., 1., 2., 3., 4.});

Static vector
-------------

.. code-block:: cpp

   constexpr size_t size = 10;
   svector<float, size> x;
   svector<float, size> y({0., 1., 2., 3., 4.});

Matrix
------

.. code-block:: cpp

   matrix<float> x({2, 3});
   matrix<float> y({2, 3}, {0., 1., 2., 3., 4., 5.});

Static matrix
-------------

.. code-block:: cpp

   smatrix<float, 2, 3> x;
   smatrix<float, 2, 3> y({0., 1., 2., 3., 4., 5.});

Tensor
------
- tensor<T>({...dims...})
- tensor<T>({...dims...}, {...data...})

Static tensor
-------------
- stensor<T, dims...>()
- stensor<T, dims...>({...data...})

Special matrices
----------------

- transposed(...)
- symmetric(...)
- hermitian(...)
- lower_tr(...)
- upper_tr(...)

