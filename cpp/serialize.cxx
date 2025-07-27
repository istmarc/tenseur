#include <ten/tensor>
#include <ten/io>

int main() {

   { // Tensor node
      auto x = ten::range<ten::matrix<float>>({3, 4});
      auto node = x.node();
      std::ofstream ofs("node.ten", std::ios_base::binary);
      ten::serialize(ofs, *node.get());
      ofs.close();

      std::ifstream ifs("node.ten", std::ios_base::binary);
      using node_type = decltype(x)::node_type;
      auto y = ten::deserialize<node_type>(ifs);
      std::cout << "Shape = " << y.shape() << std::endl;
      std::cout << "Strides = " << y.strides() << std::endl;
      std::cout << "Size of the node = " << y.size() << std::endl;
      std::cout << "Data  = \n";
      for (size_t i = 0; i < y.size(); i++)
         std::cout << y[i] << std::endl;
      ifs.close();
   }

   { // Shape
      auto x = ten::range<ten::matrix<float>>({3, 4});
      auto shape = x.shape();
      std::cout << "Shape to write = " << shape << std::endl;
      std::ofstream ofs("shape.ten", std::ios_base::binary);
      ten::serialize(ofs, shape);
      ofs.close();

      std::ifstream ifs("shape.ten", std::ios_base::binary);
      using shape_type = decltype(x)::shape_type;
      auto s = ten::deserialize<shape_type>(ifs);
      std::cout << "Read shape = " << s << std::endl;
      ifs.close();
   }

   { // Strides
      auto x = ten::range<ten::matrix<float>>({3, 4});
      auto stride = x.strides();
      std::cout << "Strides to write = " << stride << std::endl;
      std::ofstream ofs("stride.ten", std::ios_base::binary);
      ten::serialize(ofs, stride);
      ofs.close();

      std::ifstream ifs("stride.ten", std::ios_base::binary);
      using stride_type = decltype(x)::stride_type;
      auto s = ten::deserialize<stride_type>(ifs);
      std::cout << "Read stride = " << s << std::endl;
      ifs.close();
   }

   { // Tensor manual serialization
      std::cout << "\n\nTensor manual\n";
      auto x = ten::range<ten::matrix<float>>({3, 4});
      std::ofstream ofs("tensor_manual.ten", std::ios_base::binary);
      ten::serialize(ofs, x);
      ofs.close();

      std::ifstream ifs("tensor_manual.ten", std::ios_base::binary);
      using tensor_type = decltype(x);
      auto y = ten::deserialize<tensor_type>(ifs);
      ifs.close();

      std::cout << y.shape() << std::endl;
      std::cout << y.strides() << std::endl;
      std::cout << y << std::endl;
   }

   { // Tensor
      std::cout << "Tensor" << std::endl;
      auto x = ten::range<ten::matrix<float>>({3, 4});
      ten::io::save(x, "tensor.ten");
      auto y = ten::io::load<decltype(x)>("tensor.ten").value();
      std::cout << "shape = " << y.shape() << std::endl;
      std::cout << "stride = " << y.strides() << std::endl;
      std::cout << "data = \n" << y << std::endl;
   }
}
