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
   auto y = zeros<float, ten::shape<2, 3, 4>>();

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

range
-----

- Create a range tensor

.. code-block:: cpp

   auto x = ten::range<ten::tensor<float>>({2, 3, 4});
   auto y = ten::range<ten::matrix<float>>({2, 3});
   auto z = ten::range<ten::vector<float>>({10}, 0.);
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

   auto x = ten::linear<ten::tensor<float>>(0., 10., {2, 3, 4});
   auto y = ten::linear<ten::matrix<float>>(0., 10., {2, 3});
   auto z = ten::linear<ten::vector<float>>(0., 10., {10});
   constexpr size_t Rank = 3;
   auto z = ten::range<float, Rank>(0., 10., {2, 3, 4});

- Create a static linear tensor

.. code-block:: cpp

   auto x = ten::range<ten::stensor<float, 2, 3, 4>>(0., 10.);
   auto y = ten::range<float, ten::shape<2, 3, 4>>(0., 10.);

