Examples
==========

Automatic differentiation
-------------------------

Using backward() to compute the gradients of a tensor with gradient information.

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>

   int main() {
      ten::matrix<float> x = ten::range<ten::matrix<float>>({2, 3}, 1., true);
      ten::vector<float> y({3}, {7.0f, 8.0f, 9.0f}, true);
      ten::vector<float> b({2}, {10.0f, 11.0f}, true);
      auto z = x * y + b;
      auto t = ten::sum(z);
      auto r = ten::cos(t);
      r.eval();
      std::cout << r.value() << std::endl;
      r.backward();
      std::cout << "The gradients\n";
      std::cout << x.grad() << std::endl;
      std::cout << y.grad() << std::endl;
      std::cout << b.grad() << std::endl;
   }


Neural networks
---------------

Simple feed forward neural network for learning XOR.

.. code-block:: cpp

   #include <ten/io>
   #include <ten/nn>
   #include <ten/optim>
   #include <ten/tensor>

   // Define a simple neural network for learning xor
   struct xor_net : ten::nn::net {
      ten::nn::sigmoid _sigmoid;
      ten::nn::batched_dense<> _lin1;
      ten::nn::batched_dense<> _lin2;

      xor_net(size_t batch, size_t in_features, size_t out_features)
          : _sigmoid(ten::nn::sigmoid()),
            _lin1(ten::nn::batched_dense<>(batch, in_features, 5)),
            _lin2(ten::nn::batched_dense<>(batch, 5, out_features)) {
         add_param("lin1_w", _lin1.weights());
         add_param("lin1_b", _lin1.bias());
         add_param("lin2_w", _lin2.weights());
         add_param("lin2_b", _lin2.bias());
      }

      auto forward(ten::matrix<float> &x) {
         auto a = _lin1(x);
         auto b = _sigmoid(a);
         auto c = _lin2(b);
         auto d = _sigmoid(c);
         return d;
      }
   };

   int main() {
      ten::matrix<float> data({4, 2},
                              {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f});
      ten::vector<float> labels({4}, {0.0f, 1.0f, 1.0f, 0.0f});

      std::cout << "Data:" << std::endl;
      std::cout << data << std::endl;

      xor_net model(4, 2, 1);

      // Print parameters
      for (auto [name, p] : model.params()) {
         std::cout << name << ":\n";
         std::cout << p.type().name() << std::endl;
      }

      // Learning
      ten::optim::sgd<> optimizer(model.params(), 0.1f);

      size_t epochs = 400;
      for (size_t epoch = 0; epoch < epochs; epoch++) {
         // Forward pass
         auto preds = model.forward(data);
         // Compute the loss
         auto loss = ten::mse(preds, labels);
         // Evaluate the loss
         loss.eval();
         // Backward the loss
         loss.backward();
         // Optimize
         optimizer.step();
         std::cout << "Epoch [" << epoch << "] Loss: [" << loss.value() << "]"
                   << std::endl;
      }

      // Test the model
      ten::matrix<float> output = model.forward(data);
      std::cout << "Model output:\n";
      std::cout << output << std::endl;
      std::cout << "True labels:\n";
      std::cout << labels << std::endl;
   }

