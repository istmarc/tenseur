Neural networks
===============

All neural network models (classes) should inherit from ``ten::nn::net`` class, defined in `ten/neural/net.hxx <https://github.com/istmarc/tenseur/blob/main/ten/neural/net.hxx>`__.

Learnable parameters should be added to the network by calling ``add_param(name, value)``.

Activation functions
--------------------

.. list-table:: Supported activation functions
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Activation function
   * - Relu y = max(0, x)
     - ten::nn::relu
   * - Leaky relu
     - ten::nn::leaky_relu
   * - Sigmoid
     - ten::nn::sigmoid

Layers
------

.. list-table:: Supported layers
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Layer
   * - Dense/Linear layer (y = w*x + b)
     - ten::nn::dense<T=float>
   * - Batched dense/linear layer (Y = X*W + B)
     - ten::nn::batched_dense<T=float>


Optimizers
----------

Optimizers are defined in `ten/optim.hxx <https://github.com/istmarc/tenseur/blob/main/ten/optim.hxx>`__ header file.

The following optimizers are supported.

.. list-table:: Supported optimizers
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Optimizer
   * - Stochastic Gradient Descent (SGD)
     - ten::optim::sgd<T=float>

TODO: Activation functions
--------------------------

.. list-table:: Activation functions to implement
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Activation function
   * - Binary step
     - ten::nn::binary_step
   * - Hyperbolic tangent
     - ten::nn::tanh
   * - Parametric rectified linear unit (Prelu)
     - ten::nn::prelu(alpha)
   * - Gaussian
     - ten::nn::gaussian
   * - Sinusoid
     - ten::nn::sinusoid
   * - Softmax
     - ten::nn::softmax



TODO: Layers
------------

.. list-table:: Layers to implement
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Layer
   * - Convolution
     - ten::nn::cnn<T=float>
   * - Recurrent neural network
     - ten::nn::rnn<T=float>
   * - LSTM
     - ten::nn::lstm<T=float>
   * - GRU
     - ten::nn::gru<T=float>

TODO: Optimizers
----------------

- The Stochastic gradient descent implementation should be improved, by adding momentum, dampening, and nesterov.

- Here's a list of optimizers to implement

.. list-table:: Optimizers to implement
   :widths: 75 25
   :header-rows: 1

   * - Description
     - Optimizer
   * - Adam
     - ten::optim::adam<T=float>

TODO: Others
------------

Current implementation doesn't support layers without bias, this should be solved.


