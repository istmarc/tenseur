/// \file Ten/Types.hxx

#ifndef TENSEUR_TEN_TYPES_HXX
#define TENSEUR_TEN_TYPES_HXX

#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

namespace ten {
/// \enum format
/// Format type
enum class StorageFormat {
   /// Dense format
   Dense,
   /// Sparse coordinate format
   SparseCoo,
   /// Diagonal format
   Diagonal,
   /// Lower triangular format
   LowerTr,
   /// Upper triangular format
   UpperTr
};

// File type
enum class FileType{
   Infer,
   Csv,
   Text
};

#ifndef TENSEUR_SIZE_TYPE
#define TENSEUR_SIZE_TYPE std::size_t
#endif

/// \typedef size_type
/// Type of the indices
using size_type = TENSEUR_SIZE_TYPE;

// Forward declaration of shape
template <size_type Dim, size_type... Rest> class Shape;

/// \enum Order
/// Storage order of a multidimentional array
enum class StorageOrder { ColMajor, RowMajor };

static constexpr StorageOrder defaultOrder = StorageOrder::RowMajor;

/// \class tensor_base
/// Base class for tensor types
class TensorBase {
 public:
   virtual ~TensorBase() {}
};

// Expr is the base class of Scalar, Tensor and Expressions (UnaryExpr and
// BinaryExpr)
template <class Derived> class Expr;

// Concept for Expression type
template <typename T>
concept isExpr = std::is_base_of_v<::ten::Expr<T>, T>;

// Scalar type
template <class T> class Scalar;

// Traits for scalar
template <class> struct isScalar : std::false_type {};
template <class T> struct isScalar<Scalar<T>> : std::true_type {};

// Forward declaration of tensor operations
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct TensorOperations;

// Forward declaration of tensor node
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct TensorNode;

// Forward declaration of tensor type
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
class RankedTensor;

template <typename> struct isTensor : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isTensor<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = true;
};

template <typename> struct isVector : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isVector<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::rank() == 1;
};

template <typename> struct isMatrix : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isMatrix<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::rank() == 2;
};

// Dynamic tensor
template <typename> struct isDynamicTensor : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isDynamicTensor<RankedTensor<Scalar, Shape, order, Allocator, Storage>> {
   static constexpr bool value = Shape::isDynamic();
};

// Static tensor
template <typename> struct isStaticTensor : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isStaticTensor<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::isStatic();
};

/// \typedef DefaultAllocator
/// Default allocator type
template <typename T> using DefaultAllocator = std::allocator<T>;

// Storage
template <class T, class Allocator> class DenseStorage;
template <class T, size_t N> class StaticDenseStorage;
// Storage traits
template <class> struct isDenseStorage : std::false_type {};
template <class T, class Allocator>
struct isDenseStorage<DenseStorage<T, Allocator>> : std::true_type {};
template <class T, size_t N>
struct isDenseStorage<StaticDenseStorage<T, N>> : std::true_type {};

// Storage of static shape
template <class> struct isStaticStorage : std::false_type {};
template <class T, size_t N>
struct isStaticStorage<StaticDenseStorage<T, N>> : std::true_type {};

/// \typedef DefaultStorage
/// Default storage type
template <class T, class Shape>
using DefaultStorage =
    std::conditional_t<Shape::isDynamic(), DenseStorage<T, DefaultAllocator<T>>,
                       StaticDenseStorage<T, Shape::staticSize()>>;

// Dense tensor
template <class> struct isDenseTensor : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isDenseTensor<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = isDenseStorage<Storage>::value;
};

// Dense vector
template <class> struct isDenseVector : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isDenseVector<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value =
       isDenseStorage<Storage>::value && Shape::rank() == 1;
};

// Dense matrix
template <class> struct isDenseMatrix : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isDenseMatrix<RankedTensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value =
       isDenseStorage<Storage>::value && Shape::rank() == 2;
};

// Vector node
template <class> struct isVectorNode : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isVectorNode<TensorNode<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::rank() == 1;
};

template <class T>
concept VectorNode = isVectorNode<T>::value;

// Matrix node
template <class> struct isMatrixNode : std::false_type {};
template <class Scalar, class Shape, StorageOrder order, class Storage,
          class Allocator>
struct isMatrixNode<TensorNode<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::rank() == 2;
};

template <class T>
concept MatrixNode = isMatrixNode<T>::value;

// Concepts
template <class A, class B>
concept SameShape =
    std::is_same_v<typename A::shape_type, typename B::shape_type>;

template <class A, class B>
concept SameStorageOrder = A::storageOrder() ==
B::storageOrder();

template <class A, class B>
concept SameStorage =
    std::is_same_v<typename A::storage_type, typename B::storage_type>;

template <class A, class B>
concept SameAllocator =
    std::is_same_v<typename A::allocator_type, typename B::allocator_type>;

template <class A, class B>
concept SameTensor = std::same_as<A, B>;

////////////////////////////////////////////////////////////////////////////////
// Node types
// template<class> struct Node;

// Scalar node
template <class T> class ScalarNode;

// ScalarNode traits
template <class> struct isScalarNode : std::false_type {};
template <class T> struct isScalarNode<ScalarNode<T>> : std::true_type {};

// TensorNode traits
template <class> struct isTensorNode : std::false_type {};
template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct isTensorNode<TensorNode<T, Shape, Order, Storage, Allocator>>
    : std::true_type {};

// Unary Node
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class UnaryNode;

template <class> struct isUnaryNode : std::false_type {};
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
struct isUnaryNode<UnaryNode<Input, Output, Func, Args...>> : std::true_type {};

// Unary Expr
template <class E, template <typename...> class Func, typename... Args>
class UnaryExpr;

template <class> struct isUnaryExpr : std::false_type {};
template <class E, template <typename...> class Func, typename... Args>
struct isUnaryExpr<UnaryExpr<E, Func, Args...>> : std::true_type {};

// Binary Node
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
class BinaryNode;

template <class> struct isBinaryNode : std::false_type {};
template <class L, class R, class O, template <typename...> class Func,
          typename... Args>
struct isBinaryNode<BinaryNode<L, R, O, Func, Args...>> : std::true_type {};

// Binary Expr
template <class L, class R, template <typename...> class Func, typename... Args>
class BinaryExpr;

template <class> struct isBinaryExpr : std::false_type {};
template <class L, class R, template <typename...> class Func, typename... Args>
struct isBinaryExpr<BinaryExpr<L, R, Func, Args...>> : std::true_type {};

namespace traits {
// Concepts
template <typename T>
concept ScalarNode = isScalarNode<T>::value;

template <typename T>
concept TensorNode = isTensorNode<T>::value;
} // namespace traits

// Binary operation
enum class BinaryOperation { add, sub, div, mul };

} // namespace ten

namespace ten::stack {
// Enable static tensor optimizations
// Allocate static tensor of size known at compile time on the stack
#ifndef TENSEUR_ENABLE_STACK_ALLOC
#define TENSEUR_ENABLE_STACK_ALLOC true
#endif
static constexpr bool enableStackAlloc = TENSEUR_ENABLE_STACK_ALLOC;

// TODO Set default maximum size of the stack in bytes
#ifndef TENSEUR_MAX_STACK_SIZE
#define TENSEUR_MAX_STACK_SIZE 400
#endif
static constexpr size_type maxStackSize = TENSEUR_MAX_STACK_SIZE;
} // namespace ten::stack

// Debug variable
namespace ten {
#ifndef TENSEUR_DEBUG
#define TENSEUR_DEBUG false
#endif
static bool Debug = TENSEUR_DEBUG;
} // namespace ten

#endif
