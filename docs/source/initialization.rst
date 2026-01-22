Initialization
===============

zeros
-----

- Create a tensor of zeros

.. code-block:: cpp

   auto x = ten::zeros<ten::tensor<float>>({2, 3, 4});
   auto y = ten::zeros<ten::vector<float>>({10});
   auto z = ten::zeros<float, ten::shape<ten::dynamic, ten::dynamic, ten::dynamic>>({2, 3, 4});
   constexpr size_t Rank = 3;
   auto t = ten::zeros<float, Rank>({2, 3, 4});

- Create a static tensor of zeros

.. code-block:: cpp

   auto x = ten::zeros<ten::stensor<float, 2, 3, 4>>();
   auto y = ten::zeros<float, ten::shape<2, 3, 4>>();

ones
----

- Create a tensor of ones

.. code-block:: cpp

   auto x = ten::ones<ten::tensor<float>>({2, 3, 4});
   auto y = ten::ones<ten::vector<float>>({10});
   auto z = ten::ones<float, ten::shape<ten::dynamic, ten::dynamic, ten::dynamic>>({2, 3, 4});
   constexpr size_t Rank = 3;
   auto t = ten::ones<float, Rank>({2, 3, 4});

- Create a static tensor of ones

.. code-block:: cpp

   auto x = ten::ones<ten::stensor<float, 2, 3, 4>>();
   auto y = ten::ones<float, ten::shape<2, 3, 4>>();

fill
-----

- Create a tensor of filled with a single value

.. code-block:: cpp

   auto x = ten::fill<ten::tensor<float>>({2, 3, 4}, 1.0f);
   auto y = ten::fill<ten::vector<float>>({10}, 1.0f);
   auto z = ten::fill<float, ten::shape<ten::dynamic, ten::dynamic, ten::dynamic>>({2, 3, 4}, 1.0f);
   constexpr size_t Rank = 3;
   auto t = ten::fill<float, Rank>({2, 3, 4}, 1.0f);

- Create a static tensor of filled with a single value

.. code-block:: cpp

   auto x = ten::fill<ten::stensor<float, 2, 3, 4>>(1.0f);
   auto y = ten::fill<float, ten::shape<2, 3, 4>>(1.0f);

range
-----

- Create a range tensor

.. code-block:: cpp

   auto x = ten::range<ten::tensor<float>>({2, 3, 4});
   auto y = ten::range<ten::matrix<float>>({2, 3});
   auto z = ten::range<ten::vector<float>>({10});
   constexpr size_t Rank = 3;
   auto z = ten::range<float, Rank>({2, 3, 4});

- Create a static range tensor

.. code-block:: cpp

   auto x = ten::range<ten::stensor<float, 2, 3, 4>>();
   auto y = ten::range<float, ten::shape<2, 3, 4>>();

linear
------

- Create a linear tensor

.. code-block:: cpp

   auto x = ten::linear<ten::tensor<float>>({2, 3, 4}, 0., 10.);
   auto y = ten::linear<ten::matrix<float>>({2, 3}, 0., 10.);
   auto z = ten::linear<ten::vector<float>>({10}, 0., 10.);
   constexpr size_t Rank = 3;
   auto z = ten::range<float, Rank>({2, 3, 4}, 0., 10.);

- Create a static linear tensor

.. code-block:: cpp

   auto x = ten::range<ten::stensor<float, 2, 3, 4>>(0., 10.);
   auto y = ten::range<float, ten::shape<2, 3, 4>>(0., 10.);

TODO
----

- Logarithmic

- Geometric

