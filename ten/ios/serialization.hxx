#ifndef TENSEUR_IOS_SERIALIZATION
#define TENSEUR_IOS_SERIALIZATION

#include <fstream>
#include <ten/tensor>

namespace ten::io {

// FIXME Add support for serialization of const tensors
template <class Tensor>
   requires(::ten::is_tensor<Tensor>::value)
bool save(Tensor &t, std::string filename) {
   size_t index = filename.rfind(".");
   if (index == -1) {
      filename.append(".ten");
   }
   auto ext = filename.substr(index + 1, filename.size());
   if (ext != "ten") {
      std::cerr << "Unsupported file extension, please use .ten extension\n";
      return false;
   }
   std::ofstream ofs(filename, std::ios_base::binary);
   ten::serialize(ofs, t);
   ofs.close();
   return ofs.good();
}

template <class Tensor>
   requires(::ten::is_tensor<Tensor>::value)
std::optional<Tensor> load(const std::string &filename) {
   size_t index = filename.rfind(".");
   if (index == -1) {
      std::cerr << "Unsupported file type\n";
      return std::nullopt;
   }
   auto ext = filename.substr(index + 1, filename.size());
   if (ext != "ten") {
      std::cerr << "Unsupported file type\n";
      return std::nullopt;
   }
   std::ifstream ifs(filename, std::ios_base::binary);
   Tensor t = ten::deserialize<Tensor>(ifs);
   ifs.close();
   return t;
}

} // namespace ten::io

#endif
