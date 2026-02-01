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
         } else if constexpr (ten::is_matrix_v<ftype> &&
                              ten::is_matrix_v<gtype>) {
            size_t rowsf = gradf.shape().dim(0);
            size_t colsf = gradf.shape().dim(1);
            size_t rowsg = gradg.shape().dim(0);
            size_t colsg = gradg.shape().dim(1);
            bool same_values = true;
            auto first_value = gradg[0];
            for (size_t i = 0; i < gradg.size(); i++) {
               if (std::abs(first_value - gradg[i]) > 1e-6) {
                  same_values = false;
                  break;
               }
            }
            // The gradient has the same values
            if (same_values) {
               for (size_t i = 0; i < gradf.size(); i++) {
                  gradf[i] *= first_value;
               }
               // [rows x 1] x [rows x 1]
            } else if (rowsf == rowsg && colsf == 1 && colsg == 1) {
               for (size_t i = 0; i < rowsf; i++) {
                  gradf[i] *= gradg[i];
               }
               // [rowsf x 1] x [rowsg x 1] with rowsf > rowsg and rowsf % rowsg
               // == 0
            } else if (rowsf > rowsg && colsf == 1 && colsg == 1 &&
                       (rowsf % rowsg == 0)) {
               size_t j = 0;
               for (size_t i = 0; i < rowsf; i++) {
                  if (j >= rowsg) {
                     j = 0;
                  }
                  gradf[i] *= gradg[j];
                  j++;
               }
               // [rowsf x 1] x [rowsg x 1] with rowsf > rowsg and rowsf % rowsg
               // != 0
            } else if (rowsf > rowsg && colsf == 1 && colsg == 1 &&
                       (rowsf % rowsg != 0)) {
               using value_type = gtype::value_type;
               value_type mean = 0;
               for (size_t i = 0; i < gradg.size(); i++) {
                  mean += gradg[i];
               }
               mean /= value_type(gradg.size());
               for (size_t i = 0; i < gradf.size(); i++) {
                  gradf[i] *= mean;
               }
               // [rows x colsf] x [rows x 1] with colsf > 1
            } else if (rowsf == rowsg && colsf > 1 && colsg == 1) {
               for (size_t i = 0; i < colsf; i++) {
                  for (size_t j = 0; j < rowsf; j++) {
                     gradf(j, i) *= gradg[j];
                  }
               }
               // [rows x colsf] x [rows x colsg] with colsf < colsg
            } else if (rowsf == rowsg && colsf < colsg) {
               for (size_t i = 0; i < colsf; i++) {
                  for (size_t j = 0; j < rowsf; j++) {
                     gradf(j, i) *= gradg[j];
                  }
               }
               // [rowsf x cols] x [rowsg x cols] with rowsf < rowsg
            } else if (colsf == colsg && rowsf < rowsg) {
               using value_type = gtype::value_type;
               value_type mean = 0;
               for (size_t i = 0; i < gradg.size(); i++) {
                  mean += gradg[i];
               }
               mean /= value_type(gradg.size());
               for (size_t i = 0; i < gradf.size(); i++) {
                  gradf[i] *= mean;
               }
            } else {
               std::cerr << "backward_chain(matrix, matrix) no implemented\n";
            }
         } else {
            std::cerr << "backward_chain gradient not implemented\n";
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
