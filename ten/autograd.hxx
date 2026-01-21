#ifndef TENSEUR_AUTOGRAD
#define TENSEUR_AUTOGRAD

#include <ten/functional_types.hxx>
#include <ten/types.hxx>
#include <type_traits>

namespace ten {

// Backward gradient using chain rule
template <class GradientF, class GradientG>
void backward_chain(GradientF &gradf, GradientG &gradg) {
   using ftype = std::remove_cvref_t<GradientF>;
   using gtype = std::remove_cvref_t<GradientG>;
   // scalar, scalar
   if constexpr (ten::is_scalar_v<ftype> && ten::is_scalar_v<gtype>) {
      gradf.value() *= gradg.value();
   }
   // tensor, scalar
   if constexpr (ten::is_tensor_v<ftype> && ten::is_scalar_v<gtype>) {
      for (size_t i = 0; i < gradf.size(); i++) {
         gradf[i] *= gradg.value();
      }
   }
   // tensor, tensor
   if constexpr (ten::is_tensor_v<ftype> && ten::is_tensor_v<gtype>) {
      if (gradf.size() == gradg.size()) {
         for (size_t i = 0; i < gradf.size(); i++) {
            gradf[i] *= gradg[i];
         }
      } else {
         if constexpr (ten::is_vector_v<ftype> && ten::is_vector_v<gtype>) {
            size_t size = gradf.size();
            if (size % gradg.size() == 0) {
               size_t n = size / gradg.size();
               for (size_t i = 0; i < n; i++) {
                  gradf[i * n + i] *= gradg[i];
               }
            } else {
               auto val = gradg[0];
               bool same_values = true;
               for (size_t i = 0; i < gradg.size(); i++) {
                  if (std::abs(val - gradg[i]) > 1e-6) {
                     same_values = false;
                     break;
                  }
               }
               if (same_values) {
                  for (size_t i = 0; i < gradf.size(); i++) {
                     gradf[i] *= val;
                  }
               } else {
                  std::cerr << "backward_chain(vector, vector) cannot be "
                               "broadcasted\n";
               }
            }
         } else if constexpr (ten::is_matrix_v<ftype> &&
                              ten::is_vector_v<gtype>) {
            size_t rows = gradf.shape().dim(0);
            size_t cols = gradf.shape().dim(1);
            if (rows == gradg.size()) {
               for (size_t i = 0; i < cols; i++) {
                  for (size_t j = 0; j < rows; j++) {
                     gradf(j, i) *= gradg[j];
                  }
               }
            } else {
               std::cerr << "backward_chain(matrix,vector) gradient cannot be "
                            "broadcasted\n";
            }
         } else {
            std::cerr << "backward_chain gradient not implemented" << std::endl;
         }
      }
   }
   // scalar, tensor
   if constexpr (ten::is_scalar_v<ftype> && ten::is_tensor_v<gtype>) {
      for (size_t i = 0; i < gradf.size(); i++) {
         gradf.value() *= gradg[i];
      }
   }
}

} // namespace ten

#endif
