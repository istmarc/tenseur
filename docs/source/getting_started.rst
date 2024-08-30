Getting started
===============

API
---
Tenseur has an easy to use api, we illustrate it with the following example:

.. code-block:: cpp

   using namespace ten;
   normal norm;
   vector<float> a = norm.sample(1000);
   matrix<float> b;


Constructors
------------

The following constructors are defined for vectors, matrices, and tensors:

Vector
------
- vector<T>({size})
- vector<T>({size}, {...data...})

Static vector
-------------
- svector<T, size>()
- svector<T, size>({...data...})

Matrix
------
- matrix<T>({rows, cols})
- matrix<T>({rows, cols}, {...data...})

Static matrix
-------------
- smatrix<T, rows, cols>()
- smatrix<T, rows, cols>({...data...})

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

