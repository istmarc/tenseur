#ifndef TENSEUR_UNTYPED_TENSOR
#define TENSEUR_UNTYPED_TENSOR

#include <Ten/Types.hxx>
#include <Ten/Tensor.hxx>

#include <memory>
#include <type_traits>
#include <initializer_list>

namespace ten{

enum class DType{
   Float,
   Double
};

namespace details{
}

/// Tensor with no type information
template<size_type Rank>
class UntypedTensor{
private:
   // Underlying tensor type
   std::unique_ptr<TensorBase> _tensor = nullptr;
   // Type
   DType _type = DType::Float;

public:
   UntypedTensor(std::initializer_list<size_type>&& dims,
      DType type = DType::Float) :
      _type(type) {
      size_t size = 1;
      auto ptr = new Tensor<float, Rank>();
      _tensor.reset(static_cast<TensorBase*>(ptr));
   }

   UntypedTensor() {}

   template<class T>
   bool isOfType() const {
      switch(_type) {
         case DType::Float:
            return std::is_same_v<T, float>;
         break;
         case DType::Double:
            return std::is_same_v<T, double>;
         break;
      }
   }

   // Cast to an untyped tensor of type T
   //[[nodiscard]] UntypedTensor cast(DType type) {}
};

using UVector = UntypedTensor<1>;
using UMatrix = UntypedTensor<2>;
using UTensor1 = UntypedTensor<1>;
using UTensor2 = UntypedTensor<2>;
using UTensor3 = UntypedTensor<3>;
using UTensor4 = UntypedTensor<4>;

}

#endif
