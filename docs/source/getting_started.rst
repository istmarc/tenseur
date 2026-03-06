Getting started
===============

Tensor
------

A ``ten::tensor<T, Rank>`` or ``stensor<T, Dims...>`` aka ``ten::ranked_tensor<T, Shape, Order, Storage, Allocator>`` is a multidimensional array. The number of dimensions is unlimited.


By default a tensor is colum major that is its data is in a contiguous column. Support for row major tensors is limited.

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   // Access indices with () operator
   x(0, 0, 1) = 1.f;
   // Access linear indices with [] operator
   x[0] = 1.f;
   x[23] = 1.f;

Examples of a static tensor:

.. code-block:: cpp

   ten::stensor<float, 2, 3, 4> x;
   // Access indices with () operator
   x(0, 0, 1) = 1.f;
   // Access linear indices with [] operator
   x[0] = 1.f;
   x[23] = 1.f;


The number of dimensions of a tensor is returned by the function ``rank()``. The ``nth`` dimension is returned by ``dim(n)`` and the size of the tensor is returned by ``size()``.

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   std::cout << x.rank() << std::endl; // 3
   std::cout << x.dim(0) << std::endl; // 2
   std::cout << x.dim(1) << std::endl; // 3
   std::cout << x.dim(2) << std::endl; // 4
   std::cout << x.size() << std::endl; // 2*3*4 = 24


By default most functions don't copy the tensor, it can be copied by calling ``copy()``.

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   auto y = x.copy();

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

Serialization
-------------

A tensor can be saved in binary format (extension .ten) or in matrix market format.

Example:

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

