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
   if constexpr (ten::is_scalar_v<ftype> && ten::is_scalar_v<gtype>) {
      gradf.value() *= gradg.value();
   }
   if constexpr (ten::is_tensor_v<ftype> && ten::is_scalar_v<gtype>) {
      for (size_t i = 0; i < gradf.size(); i++) {
         gradf[i] *= gradg.value();
      }
   }
   if constexpr (ten::is_tensor_v<ftype> && ten::is_tensor_v<gtype>) {
      for (size_t i = 0; i < gradf.size(); i++) {
         gradf[i] *= gradg[i];
      }
   }
}

} // namespace ten

#endif
