/// \file Ten/Types.hxx

#ifndef TENSEUR_TEN_TYPES_HXX
#define TENSEUR_TEN_TYPES_HXX

#include <complex>
#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

namespace ten {
/// \enum format
/// Format type
enum class storage_format : uint16_t {
   /// None
   none = 1,
   /// Dense format
   dense = 2,
   /// Sparse coordinate format
   coo = 4,
   /// Compressed sparse colums
   csc = 8,
   /// Compressed sparse rows
   csr = 16,
   /// Diagonal format
   diagonal = 32,
   /// Lower triangular format
   lower_tr = 64,
   /// Upper triangular format
   upper_tr = 128,
   /// Symmetric
   symmetric = 256,
   /// Transposed
   transposed = 512,
   /// Hermitian
   hermitian = 1024,
};

bool operator&(storage_format a, storage_format b) {
   return static_cast<uint16_t>(a) & static_cast<uint16_t>(b);
}

storage_format operator|(storage_format a, storage_format b) {
   return static_cast<storage_format>(static_cast<uint16_t>(a) |
                                      static_cast<uint16_t>(b));
}

void operator|=(storage_format &a, const storage_format b) {
   a = static_cast<storage_format>(static_cast<uint16_t>(a) |
                                   static_cast<uint16_t>(b));
}

#ifndef TENSEUR_SIZE_TYPE
#define TENSEUR_SIZE_TYPE std::size_t
#endif

/// \typedef size_type
/// Type of the indices
using size_type = TENSEUR_SIZE_TYPE;

/// Traits for std::complex
template <class> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

// Forward declaration of shape
template <size_type Dim, size_type... Rest> class shape;

/// \enum storage_order
/// Storage order of a multidimentional array
enum class storage_order { col_major, row_major };

static constexpr storage_order default_order = storage_order::col_major;

// Storage
template <class T, class allocator> class dense_storage;
template <class T, size_t N> class sdense_storage;

// Storage traits
template <class> struct is_dense_storage : std::false_type {};
template <class T, class Allocator>
struct is_dense_storage<dense_storage<T, Allocator>> : std::true_type {};

template <class T>
concept DenseStorage = is_dense_storage<typename T::storage_type>::value;

template <class> struct is_static_storage : std::false_type {};
template <class T, size_t N>
struct is_static_storage<sdense_storage<T, N>> : std::true_type {};

template <class T>
concept StaticStorage = is_static_storage<typename T::storage_type>::value;

/// \class tensor_base
/// Base class for tensor types
class tensor_base {
 public:
   virtual ~tensor_base() {}
};

// Expr is the base class of Scalar, Tensor and Expressions (UnaryExpr and
// BinaryExpr)
template <class Derived> class expr;

// Concept for Expression type
template <typename T>
concept is_expr = std::is_base_of_v<::ten::expr<T>, T>;

// Scalar type
template <class T> class scalar;

// Traits for scalar
template <class> struct is_scalar : std::false_type {};
template <class T> struct is_scalar<scalar<T>> : std::true_type {};

// Forward declaration of tensor operations
template <class T, class Shape, storage_order Order, class Storage,
          class Allocator>
struct tensor_operations;

// Forward declaration of tensor node
template <class T, class shape, storage_order Order, class storage,
          class allocator>
struct tensor_node;

// Forward declaration of tensor type
template <class T, class shape, storage_order order, class storage,
          class allocator>
class ranked_tensor;

template <typename> struct is_tensor : std::false_type {};
template <class Scalar, class Shape, storage_order order, class Storage,
          class Allocator>
struct is_tensor<ranked_tensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = true;
};

template <typename> struct is_vector : std::false_type {};
template <class Scalar, class Shape, storage_order order, class Storage,
          class Allocator>
struct is_vector<ranked_tensor<Scalar, Shape, order, Storage, Allocator>> {
   static constexpr bool value = Shape::rank() == 1;
};

/// Matrix (dense)
template <typename> struct is_matrix : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_matrix<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       shape::rank() == 2 && ::ten::is_dense_storage<storage>::value;
};

/// StaticMatrix (dense)
template <typename> struct is_smatrix : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_smatrix<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       shape::rank() == 2 && ::ten::is_static_storage<storage>::value;
};

// Dynamic tensor
template <typename> struct is_dtensor : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_dtensor<ranked_tensor<T, shape, order, allocator, storage>> {
   static constexpr bool value = shape::is_dynamic();
};

// Static tensor
template <typename> struct is_stensor : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_stensor<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value = shape::is_static();
};

// Dynamic vector
template <typename> struct is_dvector : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_dvector<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value = shape::is_dynamic() && shape::rank() == 1;
};

// Static vector
template <typename> struct is_svector : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_svector<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value = shape::is_static() && shape::rank() == 1;
};

/// \typedef default_allocator
/// Default allocator type
template <typename T> using default_allocator = std::allocator<T>;

// Special storages
template <class T, class allocator> class diagonal_storage;
template <class T, size_type N> class sdiagonal_storage;
// template <class T, class allocator> class lowertr_storage;
// template <class T, class allocator> class uppertr_storage;

// Storage trait for special storages
template <class> struct is_diagonal_storage : std::false_type {};
template <class T, class allocator>
struct is_diagonal_storage<diagonal_storage<T, allocator>> : std::true_type {};
/*template <class> struct is_LowerTrStorage : std::false_type {};
template <class T, class allocator>
struct is_LowerTrStorage<LowerTrStorage<T, allocator>> : std::true_type {};
template <class> struct is_UpperTrStorage : std::false_type {};
template <class T, class allocator>
struct is_UpperTrStorage<UpperTrStorage<T, allocator>> : std::true_type {};*/

// Storage of static shape
// template <class> struct isStaticStorage : std::false_type {};
// template <class T, size_t N>
// struct isStaticStorage<StaticDenseStorage<T, N>> : std::true_type {};

/// \typedef DefaultStorage
/// Default storage type
template <class T, class shape>
using default_storage =
    std::conditional_t<shape::is_dynamic(),
                       dense_storage<T, default_allocator<T>>,
                       sdense_storage<T, shape::static_size()>>;

// Dense tensor
template <class> struct is_dense_tensor : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_dense_tensor<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value = is_dense_storage<storage>::value;
};

// Dense vector
template <class> struct is_dense_vector : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_dense_vector<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       is_dense_storage<storage>::value && shape::rank() == 1;
};

// Dense matrix
template <class> struct is_dense_matrix : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_dense_matrix<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       is_dense_storage<storage>::value && shape::rank() == 2;
};

// Special matrices

template <class> struct is_sdiagonal_storage : std::false_type {};
template <class T, size_t N>
struct is_sdiagonal_storage<sdiagonal_storage<T, N>> : std::true_type {};

/// Diagonal matrix
template <class> struct is_diagonal : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_diagonal<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       is_diagonal_storage<storage>::value && shape::rank() == 2;
};

/// Static diagonal matrix
template <class> struct is_sdiagonal : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_sdiagonal<ranked_tensor<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       shape::rank() == 2 && is_sdiagonal_storage<storage>::value;
};

/*
/// Lower triangular matrix
template <class> struct isLowerTrMatrix : std::false_type {};
template <class Scalar, class shape, storage_order order, class Storage,
          class Allocator>
struct isLowerTrMatrix<ranked_tensor<Scalar, shape, order, Storage, Allocator>>
{ static constexpr bool value = isLowerTrStorage<Storage>::value &&
shape::rank() == 2;
};

/// Upper triangular matrix
template <class> struct isUpperTrMatrix : std::false_type {};
template <class Scalar, class shape, storage_order order, class Storage,
          class Allocator>
struct isUpperTrMatrix<ranked_tensor<Scalar, shape, order, Storage, Allocator>>
{ static constexpr bool value = isUpperTrStorage<Storage>::value &&
shape::rank() == 2;
};*/

////////////////////////////////////////////////////////////////////////////////
// Node types
// template<class> struct Node;

// Scalar node
template <class T> class scalar_node;

// scalar_node traits
template <class> struct is_scalar_node : std::false_type {};
template <class T> struct is_scalar_node<scalar_node<T>> : std::true_type {};

// tensor_node traits
template <class> struct is_tensor_node : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_tensor_node<tensor_node<T, shape, order, storage, allocator>>
    : std::true_type {};
//{
//   static constexpr bool value = ::ten::is_dense_storage<storage>::value;
//};

// Unary Node
template <class input, class output, template <typename...> class Func,
          typename... Args>
class unary_node;

template <class> struct is_unary_node : std::false_type {};
template <class input, class output, template <typename...> class Func,
          typename... Args>
struct is_unary_node<unary_node<input, output, Func, Args...>>
    : std::true_type {};

// Unary Expr
template <class E, template <typename...> class Func, typename... Args>
class unary_expr;

template <class> struct is_unary_expr : std::false_type {};
template <class E, template <typename...> class Func, typename... Args>
struct is_unary_expr<unary_expr<E, Func, Args...>> : std::true_type {};

// Binary Node
template <class left, class right, class output,
          template <typename...> class Func, typename... Args>
class binary_node;

template <class> struct is_binary_node : std::false_type {};
template <class left, class right, class O, template <typename...> class Func,
          typename... Args>
struct is_binary_node<binary_node<left, right, O, Func, Args...>>
    : std::true_type {};

// Binary Expr
template <class left, class right, template <typename...> class Func,
          typename... Args>
class binary_expr;

template <class> struct is_binary_expr : std::false_type {};
template <class left, class right, template <typename...> class Func,
          typename... Args>
struct is_binary_expr<binary_expr<left, right, Func, Args...>>
    : std::true_type {};

/////////////////////////////////////////////////////////////////////////////////
// Concepts

/// Unary node
template <class T>
concept UnaryNode = is_unary_node<T>::value;

/// Binary node
template <class T>
concept BinaryNode = is_binary_node<T>::value;

/// Unary or binary node
template <class T>
concept Node = UnaryNode<T> || BinaryNode<T>;

/// Diagonal matrix
template <typename T>
concept Diagonal = is_diagonal<std::remove_cvref_t<T>>::value;

/// Static diagonal matrix
template <typename T>
concept SDiagonal = is_sdiagonal<std::remove_cvref_t<T>>::value;

// Concepts

/// Scalar node
template <typename T>
concept ScalarNode = is_scalar_node<T>::value;

////////////////////////////////////////////////////////////////////////////////
/// Tensor node
template <typename T>
concept TensorNode = DenseStorage<T> && is_tensor_node<T>::value;

////////////////////////////////////////////////////////////////////////////////
// Vector node

template <class> struct is_vector_node : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_vector_node<tensor_node<T, shape, order, storage, allocator>> {
   static constexpr bool value = shape::rank() == 1;
};

template <class T>
concept VectorShape = T::shape_type::rank() == 1;

template <class T>
concept VectorNode = VectorShape<T> && TensorNode<T>;

////////////////////////////////////////////////////////////////////////////////
// Matrix shape

template <class T>
concept MatrixShape = T::shape_type::rank() == 2;

////////////////////////////////////////////////////////////////////////////////
// Matrix node
template <class> struct is_matrix_node : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_matrix_node<tensor_node<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       shape::rank() == 2 && ::ten::is_dense_storage<storage>::value;
};

template <class T>
concept MatrixNode = MatrixShape<T> && TensorNode<T>;

// Static matrix node
template <class> struct is_smatrix_node : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_smatrix_node<tensor_node<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       shape::rank() == 2 && ::ten::is_static_storage<storage>::value;
};

template <class T>
concept SMatrixNode = MatrixShape<T> && StaticStorage<T> && TensorNode<T>;

////////////////////////////////////////////////////////////////////////////////
// Special matrices node
template <class> struct is_diagonal_node : std::false_type {};
template <class T, class shape, storage_order order, class storage,
          class allocator>
struct is_diagonal_node<tensor_node<T, shape, order, storage, allocator>> {
   static constexpr bool value =
       ::ten::is_diagonal_storage<storage>::value && shape::rank() == 2;
};

template <class T>
concept diagonal_node = is_diagonal_node<T>::value;

template <class T>
concept DiagonalNode = is_diagonal_node<T>::value;

////////////////////////////////////////////////////////////////////////////////
// Concepts Same
template <class A, class B>
concept same_shape =
    std::is_same_v<typename A::shape_type, typename B::shape_type>;

template <class A, class B>
concept same_storage_order = A::storage_order() ==
B::storage_order();

template <class A, class B>
concept same_storage =
    std::is_same_v<typename A::storage_type, typename B::storage_type>;

template <class A, class B>
concept same_allocator =
    std::is_same_v<typename A::allocator_type, typename B::allocator_type>;

template <class A, class B>
concept same_tensor = std::same_as<A, B>;

////////////////////////////////////////////////////////////////////////////////
// Operations

// Binary operations
enum class binary_operation { add, sub, div, mul };
// Trigonometric operations
enum class trig_operation { sin, cos, tan };

} // namespace ten

namespace ten::stack {
// Enable static tensor optimizations
// Allocate static tensor of size known at compile time on the stack
#ifndef TENSEUR_ENABLE_STACK_ALLOC
#define TENSEUR_ENABLE_STACK_ALLOC true
#endif
static constexpr bool enable_stack_alloc = TENSEUR_ENABLE_STACK_ALLOC;

// TODO Set default maximum size of the stack in bytes
#ifndef TENSEUR_MAX_STACK_SIZE
#define TENSEUR_MAX_STACK_SIZE 400
#endif
static constexpr size_type max_stack_size = TENSEUR_MAX_STACK_SIZE;
} // namespace ten::stack

#endif
