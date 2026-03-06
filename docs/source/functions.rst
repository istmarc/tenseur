Functions
=========

The functions/operators are defined in the `ten/tensor.hxx` and in `ten/functional.hxx`.

Elementwise operations
----------------------

Operators ``+``, ``-``, ``*``, and ``/`` are defined for tensors, matrices and vectors.

.. code-block:: cpp

   ten::tensor<float, Rank> x(shape);
   ten::tensor<float, Rank> y(shape);
   auto z = x op y;

Scalar tensor operations
------------------------

Operators ``+``, ``-``, ``*``, and ``/`` are overloaded for scalar - tensor, matrix and vector operations.

.. code-block:: cpp

   ten::tensor<float, Rank> x(shape);
   auto a = 1.0f op x;
   auto b = x op 1.0f;

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   auto a = 1.0f - x;
   auto b = x - 1.0f;

- Multiplication

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   auto a = 2.0f * x;
   auto b = x * 2.0f;

- Division

.. code-block:: cpp

   ten::tensor<float, 3> x({2, 3, 4});
   auto a = 1.0f / x;
   auto b = x / 2.0f;

Expression and tensor functions
-------------------------------

.. list-table:: Expression and tensor functions
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Function
   * - Minimum
     - ten::min(expr)
   * - Maximum
     - ten::max(expr)
   * - Mean
     - ten::mean(expr)
   * - Sum of the elements
     - ten::sum(expr)
   * - Cumulative sum of a vector
     - ten::cum_sum(expr)
   * - Product of the elements
     - ten::prod(expr)
   * - Elementwise absolute value
     - ten::abs(expr)
   * - Elementwise sqrt
     - ten::sqrt(expr)
   * - Elementwise squared values
     - ten::sqr(expr)
   * - Elementwise sinus
     - ten::sin(expr)
   * - Elementwise sinh
     - ten::sinh(expr)
   * - Elementwise asin
     - ten::asin(expr)
   * - Elementwise cos
     - ten::cos(expr)
   * - Elementwise cosh
     - ten::cosh(expr)
   * - Elementwise acos
     - ten::acos(expr)
   * - Elementwise tan
     - ten::tan(expr)
   * - Elementwise tanh
     - ten::tanh(expr)
   * - Elementwise atan
     - ten::atan(expr)
   * - Elementwise exp
     - ten::exp(expr)
   * - Elementwise log
     - ten::log(expr)
   * - Elementwise log10
     - ten::log10(expr)
   * - Elementwise floor
     - ten::floor(expr)
   * - Elementwise ceil
     - ten::ceil(expr)
   * - Elementwise power
     - ten::pow(expr, n)

Tensor functions
----------------

.. list-table:: Tensor functions
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Function
   * - Tests whether all the elements are close to 0 within a tolerance eps
     - ten::all_close(tensor, eps)

TODO
----

.. list-table:: Expression and tensor functions
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Function
   * - Sum along an axis or dimension
     - ten::sum(expr, axis)
   * - Mean along an axis or dimension
     - ten::mean(expr, axis)
   * - Standard deviation
     - ten::std(expr)
   * - Standard deviation along an axis or dimension
     - ten::std(expr, axis)

.. list-table:: Matrix functions
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Function
   * - Exponential of a matrix
     - ten::matrix_exp(expr)
   * - Power of a matrix
     - ten::matrix_pow(expr, n)
   * - Cosinus of a matrix
     - ten::matrix_cos(expr)
   * - Sinus of a matrix
     - ten::matrix_sin(expr)

