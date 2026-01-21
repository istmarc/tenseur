#include <any>

#include <ten/tensor>
#include <ten/io>
#include <ten/nn>
#include <ten/optim>
#include <unordered_map>

// Define a simple neural network for learning xor
struct xor_net : ten::nn::net  {
   std::unordered_map<std::string, std::any> _params;
   ten::nn::leaky_relu _leaky_relu;
   ten::nn::sigmoid _sigmoid;
   ten::nn::batched_dense<> _lin1;
   ten::nn::batched_dense<> _lin2;

   xor_net(size_t batch, size_t in_features, size_t out_features) : 
      _sigmoid(ten::nn::sigmoid()),
      _leaky_relu(ten::nn::leaky_relu()),
      _lin1(ten::nn::batched_dense<>(batch, in_features, 5)),
      _lin2(ten::nn::batched_dense<>(batch, 5, out_features)) {
      _params["lin1_w"] = _lin1.weights();
      _params["lin1_b"] = _lin1.bias();
      _params["lin2_w"] = _lin2.weights();
      _params["lin2_b"] = _lin2.bias();
   }

   auto forward(ten::matrix<float>& x) {
      auto a = _lin1(x);
      auto b = _sigmoid(a);
      auto c = _lin2(b);
      auto d = _leaky_relu(c);
      return d;
   }

   auto params() { return _params;}
};

int main() {
   ten::matrix<float> data({4, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f});
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

   size_t epochs = 300;
   for (size_t epoch = 0; epoch < epochs; epoch++) {
      // No need for optimizer.zero_grad();
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
      std::cout << "Epoch [" << epoch << "] Loss: [" << loss.value() << "]" << std::endl;
   }

   // Test the model
   auto output = model.forward(data);
   output.eval();
   auto output_tensor = output.value();
   std::cout << "Model output:\n";
   std::cout << output_tensor << std::endl;
   std::cout << "True labels:\n";
   std::cout << labels << std::endl;
}

