#ifndef TENSUER_IOS_MTX
#define TENSUER_IOS_MTX

#include <ten/tensor.hxx>

namespace ten::io {

// Save vector to a file
template <class T>
   requires(::ten::is_vector<T>::value &&
            std::is_floating_point_v<typename T::value_type>)
void save_mtx(const T &t, std::string filename) {
   std::cout << "Tenseur: Saving vector to file " << filename << std::endl;
   if (filename.empty()) {
      return;
   }
   size_t index = filename.rfind(".");
   if (index == -1) {
      filename.append(".ext");
   }
   auto ext = filename.substr(index, filename.size());
   if (ext != ".mtx") {
      std::cout << "Unsuported file extension" << std::endl;
      return;
   }
   size_t size = t.size();

   std::ofstream file(filename, std::ios::out);
   file << "%%MatrixMarket matrix array real general\n";
   file << "%\n";
   file << size << " 1\n";

   for (size_t i = 0; i < size; i++) {
      file << t[i] << "\n";
   }
}

// Save matrix to a file
template <class T>
   requires(::ten::is_matrix<T>::value &&
            std::is_floating_point_v<typename T::value_type>)
void save_mtx(const T &t, std::string filename) {
   std::cout << "Tenseur: Saving Matrix to file " << filename << std::endl;
   if (filename.empty()) {
      return;
   }
   size_t index = filename.rfind(".");
   if (index == -1) {
      filename.append(".ext");
   }
   auto ext = filename.substr(index, filename.size());
   if (ext != ".mtx") {
      std::cout << "Unsuported file extension\n";
      return;
   }
   size_t m = t.dim(0);
   size_t n = t.dim(1);

   std::ofstream file(filename, std::ios::out);
   file << "%%MatrixMarket matrix array real general\n";
   file << "%\n";
   file << m << " " << n << "\n";

   for (size_t k = 0; k < m * n; k++) {
      file << t[k] << "\n";
   }
}

} // namespace ten::io

#endif
