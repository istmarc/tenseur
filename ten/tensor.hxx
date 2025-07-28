/// \file Ten/Tensor.hxx

#ifndef TENSEUR_TENSOR_HXX
#define TENSEUR_TENSOR_HXX

#include <algorithm>
#include <array>
#include <complex>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ten/expr.hxx>
#include <ten/functional.hxx>
#include <ten/shape.hxx>
#include <ten/types.hxx>
#include <ten/utils.hxx>

#include <ten/storage/dense_storage.hxx>
#include <ten/storage/diagonal_storage.hxx>

namespace ten {

/// \class Expr
/// Represent an expression
template <typename Derived> class expr {
 public:
   // using type = static_cast<Derived>(*this);

 protected:
   expr() = default;
   expr(const expr &) = default;
   expr(expr &&) = default;
};

// Add two expr
template <Expr LeftExpr, Expr RightExpr>
auto operator+(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::add>::func>(
       left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator+(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::add>::func>(
       ::ten::scalar<T>(scalar), expr);
}

template <Expr E, typename T>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator+(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::add>::func>(
       expr, ::ten::scalar<T>(scalar));
}

// Substract two expressions
template <Expr LeftExpr, Expr RightExpr>
auto operator-(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::sub>::func>(
       left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator-(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::sub>::func>(
       ::ten::scalar<T>(scalar), expr);
}

template <Expr E, typename T>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator-(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::sub>::func>(
       expr, ::ten::scalar<T>(scalar));
}

// Multiply two expressions
template <Expr LeftExpr, Expr RightExpr>
auto operator*(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;

   using output_type = ::ten::details::mul_result_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::mul<left_input, right_input,
                              output_type>::template func>(left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator*(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::mul>::func>(
       ::ten::scalar<T>(scalar), expr);
}
template <Expr E, typename T>
   requires(::std::is_floating_point_v<T> || ten::is_complex<T>::value ||
            std::is_integral_v<T>)
auto operator*(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::mul>::func>(
       expr, ::ten::scalar<T>(scalar));
}

// Divide two expressions
template <Expr LeftExpr, Expr RightExpr>
auto operator/(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::div>::func>(
       left, right);
}

/*
template <typename T,Expr E>
auto operator/(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return scalar<T>(scalar) / std::forward<R>(expr);
}*/

/*
template <Expr E, typename T> auto operator/(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) / ::ten::scalar<T>(scalar);
}*/

/// TODO AxpyOne
/// A += B
/*template <typename left_expr, typename right_expr>
   requires ::ten::is_expr<std::remove_cvref_t<left_expr>> &&
            ::ten::is_expr<std::remove_cvref_t<right_expr>>
void operator+=(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::Axpy>(
       left.node(), right.node());
}*/

/// \class scalar_operations
template <class T> struct scalar_operations {
   [[nodiscard]] inline static constexpr size_type rank() { return 0; }
};

/// \class scalar
/// Hold a single value of type T.
template <typename T>
class scalar : public expr<scalar<T>>, public scalar_operations<scalar<T>> {
 public:
   using node_type = scalar_node<T>;
   using value_type = T;

 private:
   T _value;

 public:
   scalar() {}

   explicit scalar(const T &value) : _value(value) {}

   explicit scalar(T &&value) : _value(std::move(value)) {}

   /// Asignment from an expression
   /*template <class Expr>
      requires(::ten::is_unary_expr<std::remove_cvref_t<Expr>>::value ||
               ::ten::is_binary_expr<std::remove_cvref_t<Expr>>::value)
   scalar(Expr &&expr) {
      _node = expr.eval();
   }*/

   /// Get the value
   const T &value() const { return _value; }

   scalar &operator=(const T &value) {
      _value = value;
      return *this;
   }

   scalar &operator=(const scalar &s) {
      _value = s._value;
      return *this;
   }

   // ostream operator
   template <class Type>
   friend std::ostream &operator<<(std::ostream &, const scalar<Type> &);
};

template <class T>
std::ostream &operator<<(std::ostream &os, const scalar<T> &s) {
   os << s.value();
   return os;
}

/// \class tensor_operations
/// Tensor operations
template <class __t, class shape, storage_order order, class storage,
          class allocator>
struct tensor_operations {
   /// \typedef value_type
   /// Value type
   using value_type = __t;

   /// \typedef scalar_type
   /// scalar type
   using scalar_type = scalar<__t>;

   /// \typedef shape_type
   /// shape type
   using shape_type = shape;

   /// \typedef storage_type
   /// Storage type
   using storage_type = storage;

   /// \typedef allocator_type
   /// __type of the allocator
   using allocator_type = allocator;

   /// \typedef stride_type
   /// Stride type
   using stride_type = stride<shape, order>;

   /// Returns the storage order
   [[nodiscard]] inline static constexpr storage_order storage_order() {
      return order;
   }

   //// Returns the rank
   [[nodiscard]] inline static constexpr size_type rank() {
      return shape::rank();
   }

   /// Returns the static size
   [[nodiscard]] inline static constexpr size_type static_size()
      requires(shape::is_static())
   {
      return shape::static_size();
   }

   /// Returns whether the tensor is of static shape
   [[nodiscard]] inline static constexpr bool is_static() {
      return shape::is_static();
   }

   /// Returns whether the tensor is of dynamic shape
   [[nodiscard]] inline static constexpr bool is_dynamic() {
      return shape::is_dynamic();
   }

   /// Returns whether the Index'th dim is static
   template <size_type index>
   [[nodiscard]] inline static constexpr bool is_static_dim() {
      return shape::template is_static_dim<index>();
   }

   /// Returns whether the Index'th dim is dynamic
   template <size_type index>
   [[nodiscard]] inline static constexpr bool is_dynamic_dim() {
      return shape::template is_dynamic_dim<index>();
   }

   /// Return the Index'th static dimension
   template <size_type index>
   [[nodiscard]] inline static constexpr size_type static_dim() {
      return shape::template static_dim<index>();
   }

   /// Returns whether the tensor is a vector
   [[nodiscard]] static constexpr bool is_vector() {
      return shape::rank() == 1;
   }

   /// Returns whether the tensor is a matrix
   [[nodiscard]] static constexpr bool is_matrix() {
      return shape::rank() == 2;
   }
};

/// \class tensor_node
/// Tensor node
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
class tensor_node
    : public tensor_operations<__t, __shape, __order, __storage, __allocator> {
 public:
   /// \typedef base_type
   /// Base type
   using base_type =
       tensor_operations<__t, __shape, __order, __allocator, __storage>;

   /// Tensor type
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;

   using scalarnode_type = scalar_node<__t>;

   using stride_type = typename base_type::stride_type;

   using storage_type = __storage;

 private:
   /// Storage format
   ::ten::storage_format _format = ::ten::storage_format::dense;
   /// Optional shape (only for dynamic tensors)
   std::optional<__shape> _shape = std::nullopt;
   /// Optional stride (only for dynamic tensors)
   std::optional<stride_type> _stride = std::nullopt;
   /// storage
   std::unique_ptr<__storage> _storage = nullptr;

 private:
   /// Returns the value at the indices
   [[nodiscard]] inline typename base_type::value_type &
   at(size_type index, auto... tail) noexcept {
      static constexpr size_type rank = __shape::rank();
      constexpr size_type tail_size = sizeof...(tail);
      static_assert(tail_size == 0 || tail_size == (rank - 1),
                    "Invalid number of indices.");
      if constexpr (tail_size == 0) {
         return (*_storage.get())[index];
      }
      std::array<size_type, __shape::rank()> indices{
          index, static_cast<size_type>(tail)...};
      if constexpr (__shape::is_dynamic()) {
         size_type idx = details::linear_index(_stride.value(), indices);
         return (*_storage.get())[idx];
      } else {
         size_type idx = details::static_linear_index<stride_type>(indices);
         return (*_storage.get())[idx];
      }
   }

   /// Returns the value at the indices
   [[nodiscard]] inline const typename base_type::value_type &
   at(size_type index, auto... tail) const noexcept {
      static constexpr size_type rank = __shape::rank();
      constexpr size_type tail_size = sizeof...(tail);
      static_assert(tail_size == 0 || tail_size == (rank - 1),
                    "Invalid number of indices.");
      if constexpr (tail_size == 0) {
         return (*_storage.get())[index];
      }
      std::array<size_type, __shape::rank()> indices{
          index, static_cast<size_type>(tail)...};
      if constexpr (__shape::is_dynamic()) {
         size_type idx = details::linear_index(_stride.value(), indices);
         return (*_storage.get())[idx];
      } else {
         size_type idx = details::static_linear_index<stride_type>(indices);
         return (*_storage.get())[idx];
      }
   }

 public:
   // For deserialization
   tensor_node(::ten::storage_format format, std::optional<__shape> &&shape,
               std::optional<stride_type> &&stride,
               std::unique_ptr<__storage> &&storage)
       : _format(format), _shape(std::move(shape)), _stride(std::move(stride)),
         _storage(std::move(storage)) {}

   /// Construct a static tensor_node
   tensor_node() noexcept
       : _shape(std::nullopt), _stride(std::nullopt),
         _storage(new __storage()) {}

   /// Construct a tensor_node from a list of shape
   explicit tensor_node(std::initializer_list<size_type> &&dims) noexcept
      requires(__shape::is_dynamic())
       : _shape(std::move(dims)), _storage(new __storage(_shape.value())),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Tensor node from shape and data
   explicit tensor_node(std::initializer_list<size_type> &&dims,
                        std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_dynamic())
       : _shape(std::move(dims)), _storage(new __storage(_shape.value())),
         _stride(typename base_type::stride_type(_shape.value())) {
      size_type i = 0;
      for (auto x : data) {
         (*_storage.get())[i] = x;
         i++;
      }
   }

   /// Construct a tensor_node from the shape
   explicit tensor_node(const __shape &dims) noexcept
      requires(__shape::is_dynamic())
       : _shape(dims), _storage(new __storage(dims)),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Tensor node from shape and data
   explicit tensor_node(const __shape &dims,
                        std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_dynamic())
       : _shape(dims), _storage(new __storage(_shape.value())),
         _stride(typename base_type::stride_type(_shape.value())) {
      size_type i = 0;
      for (auto x : data) {
         (*_storage.get())[i] = x;
         i++;
      }
   }

   /// Tensor node from data
   explicit tensor_node(std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_static())
       : _shape(std::nullopt), _stride(std::nullopt),
         _storage(new __storage()) {
      size_type i = 0;
      for (auto x : data) {
         (*_storage.get())[i] = x;
         i++;
      }
   }

   /// Construct a tensor_node from storage
   explicit tensor_node(std::unique_ptr<__storage> &&st) noexcept
      requires(__shape::is_static())
       : _storage(std::move(st)) {}

   /// Construct a tensor_node from storage and shape
   /// FIXME Check size
   explicit tensor_node(std::unique_ptr<__storage> &&st,
                        const __shape &dims) noexcept
      requires(__shape::is_dynamic())
       : _storage(std::move(st)), _shape(dims),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Construct a tenso_node from storage and format
   explicit tensor_node(std::unique_ptr<__storage> &&st,
                        ::ten::storage_format format) noexcept
      requires(__shape::is_static())
       : _format(format), _storage(std::move(st)) {}

   /// Construct a tenso_node from storage and shape
   /// FIXME Check size
   explicit tensor_node(std::unique_ptr<__storage> &&st, const __shape &dims,
                        ::ten::storage_format format) noexcept
      requires(__shape::is_dynamic())
       : _format(format), _storage(std::move(st)), _shape(dims),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Move constructor
   tensor_node(tensor_node &&) = default;

   /// Copy constructor
   tensor_node(const tensor_node &) = default;

   /// Assignment operator
   tensor_node &operator=(tensor_node &&) = default;

   /*tensor_node &operator=(const tensor_node &node) {
      _storage = node.storage();
      return *this;
   }*/
   // Assignement operator
   tensor_node &operator=(const tensor_node &node) {
      this->_format = node._format;
      this->_shape = node._shape;
      this->_stride = node._stride;
      this->_storage = node._storage;
      return *this;
   }

   /// Get the dimension at index
   [[nodiscard]] size_type dim(size_type index) const {
      return _shape.value().dim(index);
   }

   /// Get the size
   [[nodiscard]] size_type size() const {
      if constexpr (__shape::is_dynamic()) {
         return _storage.get()->size();
      } else {
         return storage_type::size();
      }
   }

   /// Get the shape
   [[nodiscard]] inline const __shape &shape() const { return _shape.value(); }

   /// Get the strides
   [[nodiscard]] inline const typename base_type::stride_type &strides() const {
      return _stride.value();
   }

   /// Get the data
   [[nodiscard]] __t *data() { return _storage.get()->data(); }

   /// Get the data
   [[nodiscard]] const __t *data() const { return _storage.get()->data(); }

   /// Get the storage format
   [[nodiscard]] storage_format format() const { return _format; }

   /// Return whether the tensor is transposed
   [[nodiscard]] bool is_transposed() const {
      return _format & ::ten::storage_format::transposed;
   }

   /// Return whether the tensor is symmetric
   [[nodiscard]] bool is_symmetric() const {
      return _format & ::ten::storage_format::symmetric;
   }

   /// Return whether the tensor is hermitian
   [[nodiscard]] bool is_hermitian() const {
      return _format & ::ten::storage_format::hermitian;
   }

   /// Return whether the tensor is diagonal
   [[nodiscard]] bool is_diagonal() const {
      return _format == ::ten::storage_format::diagonal;
   }

   /// Return whether the tensor is lower triangular
   [[nodiscard]] bool is_lower_tr() const {
      return _format == ::ten::storage_format::lower_tr;
   }

   /// Return whether the tensor is upper triangular
   [[nodiscard]] bool is_upper_tr() const {
      return _format == ::ten::storage_format::upper_tr;
   }

   /// Return whether the tensor is sparse coo
   [[nodiscard]] bool is_sparse_coo() const {
      return _format == ::ten::storage_format::coo;
   }

   /// Return whether the tensor is sparse csc
   [[nodiscard]] bool is_sparse_csc() const {
      return _format == ::ten::storage_format::csc;
   }

   /// Return whether the tensor is sparse csr
   [[nodiscard]] bool is_sparse_csr() const {
      return _format == ::ten::storage_format::csr;
   }

   /// Get the storage
   [[nodiscard]] __storage &storage() const { return *_storage.get(); }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      return at(index);
   }

   /// Overloading the [] operator
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      return at(index);
   }

   /// Overloading the () operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator()(auto... index) const noexcept {
      return at(index...);
   }

   /// Overloading the () operator
   [[nodiscard]] inline typename base_type::value_type &
   operator()(auto... index) noexcept {
      return at(index...);
   }

   template <class T, class Shape, storage_order Order, class Storage,
             class Allocator>
   friend bool
   serialize(std::ostream &os,
             tensor_node<T, Shape, Order, Storage, Allocator> &node);

   template <class TensorNode>
      requires(::ten::is_tensor_node<TensorNode>::value)
   friend TensorNode deserialize(std::istream &os);
};

template <class T, class Shape, storage_order Order, class Storage,
          class Allocator>
bool serialize(std::ostream &os,
               tensor_node<T, Shape, Order, Storage, Allocator> &node) {
   bool good = true;

   os.write(reinterpret_cast<char *>(&node._format), sizeof(node._format));
   good |= os.good();
   // Serialize shape
   if (!node._shape.has_value()) {
      std::cerr << "Seriaization of this type of shape not yet supported.\n";
      return false;
   }
   good |= serialize(os, node._shape.value());
   // Serialize stride
   if (!node._stride.has_value()) {
      std::cerr << "Seriaization of this type of stride not yet supported.\n";
      return false;
   }
   good |= serialize(os, node._stride.value());

   // Serialize storage
   Storage *storage_pointer = node._storage.get();
   if (storage_pointer == nullptr) {
      std::cerr << "Error: Serialization of null storage pointer\n";
      return false;
   }
   return good | serialize(os, *storage_pointer);
}

template <class TensorNode>
   requires(::ten::is_tensor_node<TensorNode>::value)
TensorNode deserialize(std::istream &is) {
   ::ten::storage_format format;
   is.read(reinterpret_cast<char *>(&format), sizeof(format));

   using Shape = typename TensorNode::shape_type;
   Shape shape_value = deserialize<Shape>(is);
   std::optional<Shape> shape(shape_value);

   using stride_type = typename TensorNode::stride_type;
   stride_type stride_value = deserialize<stride_type>(is);
   std::optional<stride_type> stride(stride_value);

   using Storage = typename TensorNode::storage_type;
   auto st = std::unique_ptr<Storage>(new Storage(deserialize<Storage>(is)));

   return TensorNode(format, std::move(shape), std::move(stride),
                     std::move(st));
}

namespace details {
/// Type of the allocator
template <typename __storage> struct allocator_type {
   using type = typename __storage::allocator_type;
};
} // namespace details

/// \class Tensor
///
/// Tensor represented by a multidimentional array.
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
class ranked_tensor final
    : public ::ten::expr<
          ranked_tensor<__t, __shape, __order, __storage, __allocator>>,
      public ::ten::tensor_operations<__t, __shape, __order, __storage,
                                      __allocator>,
      public ::ten::tensor_base {
 public:
   /// \typedef casted_type
   /// Type of the casted tensor
   template <typename __to>
      requires std::convertible_to<__t, __to>
   using casted_type =
       ranked_tensor<__to, __shape, __order,
                     typename __storage::template casted_type<__to>,
                     typename details::allocator_type<
                         typename __storage::template casted_type<__to>>::type>;

   /// \typedef base_type
   /// Type of the tensor operations.
   using base_type =
       tensor_operations<__t, __shape, __order, __storage, __allocator>;

   /// \typedef node_type
   /// Node type
   using node_type = tensor_node<__t, __shape, __order, __storage, __allocator>;

 private:
   /// Shared pointer to the node
   // TODO Slice / View / or TilledTensor
   std::shared_ptr<node_type> _node = nullptr;

 public:
   /// Constructor for static tensor
   ranked_tensor() noexcept : _node(std::make_shared<node_type>()) {}

   /// Constructor for tensor with a storage of type ten::dense_storage
   explicit ranked_tensor(std::initializer_list<size_type> &&dims) noexcept
      requires(__shape::is_dynamic())
       : _node(std::make_shared<node_type>(std::move(dims))) {}

   /// Constructor of tensor from shape
   explicit ranked_tensor(const __shape &dims) noexcept
       : _node(std::make_shared<node_type>(dims)) {}

   /// Construct a tensor from shape and data
   explicit ranked_tensor(const __shape &dims,
                          std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_dynamic())
       : _node(std::make_shared<node_type>(dims, std::move(data))) {}

   /// Construct a tensor from shape and data
   explicit ranked_tensor(std::initializer_list<size_type> &&dims,
                          std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_dynamic())
       : _node(std::make_shared<node_type>(std::move(dims), std::move(data))) {}

   /// Constructor for static tensor from data
   explicit ranked_tensor(std::initializer_list<__t> &&data) noexcept
      requires(__shape::is_static())
       : _node(std::make_shared<node_type>(std::move(data))) {}

   /// Constructor of tensor from a shared pointer to tensor_node
   ranked_tensor(const std::shared_ptr<node_type> &node) : _node(node) {}

   /// Constructor of tensor from a shared pointer to tensor_node
   ranked_tensor(std::shared_ptr<node_type> &&node) : _node(std::move(node)) {}

   /// Asignment from a static expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_static())
   ranked_tensor(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Error: Evaluated type must be a tensor.");

      auto node = expr.eval().node();

      static_assert(__shape::static_size() == evaluated_type::static_size(),
                    "Expected equal shape size.");
      _node = node;
   }

   /// Asignment from a dynamic expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_dynamic())
   ranked_tensor(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Evaluated type must be a tensor.");

      auto node = expr.eval().node();
      _node = node;
   }

   /// Constructor for vector
   explicit ranked_tensor(size_type size) noexcept
      requires(__shape::is_dynamic() && __shape::rank() == 1)
   {
      _node =
          std::make_shared<node_type>(std::initializer_list<size_type>{size});
   }

   /// Constructor for matrix
   explicit ranked_tensor(size_type rows, size_type cols) noexcept
      requires(__shape::is_dynamic() && __shape::rank() == 2)
   {
      _node = std::make_shared<node_type>(
          std::initializer_list<size_type>{rows, cols});
   }

   // TODO CooMatrix

   // TODO CscMatrix

   // TODO CsrMatrix

   /// Copy constructor
   ranked_tensor(const ranked_tensor &t) { _node = t._node; }

   /// Copy constructor
   ranked_tensor(ranked_tensor &&t) { _node = std::move(t._node); }

   /// Assignment operator
   ranked_tensor &operator=(const ranked_tensor &t) {
      _node = t._node;
      return *this;
   }

   /// Assignment operator
   ranked_tensor &operator=(ranked_tensor &&t) {
      _node = std::move(t._node);
      return *this;
   }

   // TODO Iterators

   /// Returns the shape
   [[nodiscard]] inline const __shape &shape() const {
      return _node.get()->shape();
   }

   /// Returns the strides
   [[nodiscard]] inline const typename base_type::stride_type &strides() const {
      return _node.get()->strides();
   }

   /// Returns the dynamic size
   [[nodiscard]] inline size_type size() const { return _node.get()->size(); }

   /// Returns the index'th dynamic dimension
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _node.get()->dim(index);
   }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Get the data
   [[nodiscard]] const __t *data() const { return _node.get()->data(); }

   /// Get the data
   [[nodiscard]] __t *data() { return _node.get()->data(); }

   /// Returns the storage
   [[nodiscard]] __storage &storage() const { return _node.get()->storage(); }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      return (*_node.get())[index];
   }

   /// Overloading the [] operator
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      return (*_node.get())[index];
   }

   /// Overloading the () operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator()(auto... index) const noexcept {
      return (*_node.get())(index...);
   }

   /// Overloading the () operator
   [[nodiscard]] inline typename base_type::value_type &
   operator()(auto... index) noexcept {
      return (*_node.get())(index...);
   }

   // Get the column
   [[nodiscard]] auto column(const size_t index) const -> decltype(auto)
      requires(__shape::rank() == 2)
   {
      return ranked_column<__t, __shape, __order, __storage, __allocator>(
          _node, index);
   }

   // Get the row
   [[nodiscard]] auto row(const size_t index) -> decltype(auto)
      requires(__shape::rank() == 2)
   {
      return ranked_row<__t, __shape, __order, __storage, __allocator>(_node,
                                                                       index);
   }

   // copy
   auto copy() const {
      auto format = _node.get()->format();
      if constexpr (__shape::is_dynamic()) {
         auto shape = _node.get()->shape();
         auto st = std::make_shared<typename node_type::storage_type>(shape);
         auto node = std::make_shared<node_type>(std::move(st), shape, format);
         ranked_tensor t(std::move(node));
         // Copy the data to the new tensor
         for (size_t i = 0; i < this->size(); i++) {
            t[i] = (*_node.get())[i];
         }
         return t;
      } else {
         auto st = std::make_shared<typename node_type::storage_type>();
         auto node = std::make_shared<node_type>(std::move(st), format);
         ranked_tensor t(std::move(node));
         // Copy the data to the new tensor
         for (size_t i = 0; i < this->size(); i++) {
            t[i] = (*_node.get())[i];
         }
         return t;
      }
   }

   /// Get the storage format
   [[nodiscard]] storage_format format() const { return _node.get()->format(); }

   /// Returns whether the tensor is transposed
   [[nodiscard]] bool is_transposed() const {
      return _node.get()->is_transposed();
   }

   /// Returns whether the tensor is symmetric
   [[nodiscard]] bool is_symmetric() const {
      return _node.get()->is_symmetric();
   }

   /// Returns whether the tensor is hermitian
   [[nodiscard]] bool is_hermitian() const {
      return _node.get()->is_hermitian();
   }

   /// Returns whether the tensor is diagonal
   [[nodiscard]] bool is_diagonal() const { return _node.get()->is_diagonal(); }

   /// Returns whether the tensor is lower triangular
   [[nodiscard]] bool is_lower_tr() const { return _node.get()->is_lower_tr(); }

   /// Returns whether the tensor is upper triangular
   [[nodiscard]] bool is_upper_tr() const { return _node.get()->is_upper_tr(); }

   /// Returns whether the tensor is sparse coordiante format
   [[nodiscard]] bool is_sparse_coo() const {
      return _node.get()->is_sparse_coo();
   }

   /// Returns whether the tensor is compressed sparse colums
   [[nodiscard]] bool is_sparse_csc() const {
      return _node.get()->is_sparse_csc();
   }

   /// Returns whether the tensor is compressed sparse rows
   [[nodiscard]] bool is_sparse_csr() const {
      return _node.get()->is_sparse_csr();
   }

   /// Returns whether the tensor is sparse
   [[nodiscard]] bool is_sparse() const {
      return _node.get()->is_sparse_coo() || _node.get()->is_sparse_csc() ||
             _node.get()->is_sparse_csr();
   }

   // Data type
   [[nodiscard]] inline ten::data_type data_type() noexcept {
      return to_data_type<__t>();
   }

   /// std::ostream friend function
   template <class __type, class __shape_type, storage_order __storage_order,
             class __storage_type, class __allocator_type>
   friend std::ostream &
   operator<<(std::ostream &os,
              const ranked_tensor<__type, __shape_type, __storage_order,
                                  __storage_type, __allocator_type> &t);

   // Serialize friend function
   template <class T, class Shape, storage_order StorageOder, class Storage,
             class Allocator>
   friend bool
   serialize(std::ostream &os,
             ranked_tensor<T, Shape, StorageOder, Storage, Allocator> &t);

   // Deserialize friend function
   template <class TensorType>
      requires(ten::is_tensor<TensorType>::value)
   friend TensorType deserialize(std::istream &is);
};

template <class T, class Shape, storage_order StorageOder, class Storage,
          class Allocator>
bool serialize(std::ostream &os,
               ranked_tensor<T, Shape, StorageOder, Storage, Allocator> &t) {
   return serialize(os, *t._node.get());
}

template <class TensorType>
   requires(ten::is_tensor<TensorType>::value)
TensorType deserialize(std::istream &is) {
   using NodeType = TensorType::node_type;
   auto node =
       std::shared_ptr<NodeType>(new NodeType(deserialize<NodeType>(is)));
   return TensorType(std::move(node));
}

/// Overload << operator for vector
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_dvector<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "vector<" << ::ten::to_string<__t>() << "," << t.shape() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for a static vector
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_svector<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "svector<" << ::ten::to_string<__t>() << "," << __shape::static_size()
      << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/// Overload << operator for matrix
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_matrix<ranked_tensor<__t, __shape, __order, __storage,
                                           __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "matrix<" << ::ten::to_string<__t>() << "," << t.shape() << ">";
   size_type m = t.dim(0);
   size_type n = t.dim(1);
   for (size_type i = 0; i < m; i++) {
      os << "\n";
      os << t(i, 0);
      for (size_type j = 1; j < n; j++) {
         os << "   " << t(i, j);
      }
   }
   return os;
}

/// Overload << operator for static matrix
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_smatrix<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "smatrix<" << ::ten::to_string<__t>() << ","
      << __shape::template static_dim<0>() << "x"
      << __shape::template static_dim<1>() << ">";
   size_type m = __shape::template static_dim<0>();
   size_type n = __shape::template static_dim<1>();
   for (size_type i = 0; i < m; i++) {
      os << "\n";
      os << t(i, 0);
      for (size_type j = 1; j < n; j++) {
         os << "   " << t(i, j);
      }
   }
   return os;
}

/// Overload << operator for diagonal matrix
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_diagonal<ranked_tensor<__t, __shape, __order, __storage,
                                             __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "diagonal<" << ::ten::to_string<__t>() << "," << t.shape() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < size; i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "⋮\n";
      for (size_type i = size - 5; i < size; i++) {
         os << "\n" << t[i];
      }
   }

   return os;
}

/// Overload << operator for static diagonal matrix
template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
   requires(::ten::is_sdiagonal<ranked_tensor<__t, __shape, __order, __storage,
                                              __allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<__t, __shape, __order, __storage, __allocator> &t) {
   os << "sdiagonal<" << ::ten::to_string<__t>() << ","
      << __shape::template static_dim<0>() << "x"
      << __shape::template static_dim<1>() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < size; i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "⋮\n";
      for (size_type i = size - 5; i < size; i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

// vector<T>
template <class T, storage_order Order = default_order,
          class Storage = default_storage<T, ::ten::shape<::ten::dynamic>>,
          class Allocator = typename details::allocator_type<Storage>::type>
using vector =
    ranked_tensor<T, shape<::ten::dynamic>, Order, Storage, Allocator>;

// svector<T, size>
template <class T, size_type Size, storage_order Order = default_order,
          class Storage = default_storage<T, ::ten::shape<Size>>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(Size > 0)
using svector = ranked_tensor<T, shape<Size>, Order, Storage, Allocator>;

// matrix<T> or matrix<T, Shape>
template <class T, class Shape = dynamic_shape<2>,
          storage_order Order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(Shape::rank() == 2)
using matrix = ranked_tensor<T, Shape, Order, Storage, Allocator>;

// smatrix<T, rows, cols>
template <class T, size_type Rows, size_type Cols,
          storage_order Order = default_order,
          class Storage = default_storage<T, ::ten::shape<Rows, Cols>>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(Rows > 0 && Cols > 0)
using smatrix = ranked_tensor<T, shape<Rows, Cols>, Order, Storage, Allocator>;

/// \typedef tensor
/// tensor<T, Rank>
template <class T, size_type Rank, storage_order Order = default_order,
          class Storage = default_storage<T, dynamic_shape<Rank>>,
          class Allocator = typename details::allocator_type<Storage>::type>
using tensor = ranked_tensor<T, dynamic_shape<Rank>, Order, Storage, Allocator>;

/// \typedef stensor
/// stensor<T, Dims...>
template <class T, size_type... Dims>
   requires(shape<Dims...>::is_static())
using stensor = ranked_tensor<
    T, shape<Dims...>, default_order, default_storage<T, shape<Dims...>>,
    typename details::allocator_type<default_storage<T, shape<Dims...>>>::type>;

////////////////////////////////////////////////////////////////////////////////
// Special matrices

// diagonal<__t>
template <class __t, class __shape = dynamic_shape<2>,
          storage_order __order = default_order,
          class __storage = diagonal_storage<__t, default_allocator<__t>>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(__shape::rank() == 2)
using diagonal = ranked_tensor<__t, __shape, __order, __storage, __allocator>;

// sdiagonal<__t, rows, cols>
/// TODO Extend to diagonal triangular matrices
template <class __t, size_type rows, size_type cols,
          storage_order __order = default_order,
          class __storage = sdiagonal_storage<__t, rows>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(rows == cols)
using sdiagonal =
    ranked_tensor<__t, ten::shape<rows, cols>, __order, __storage, __allocator>;

////////////////////////////////////////////////////////////////////////////////

/// transposed matrix
template <class T>
   requires(T::is_dynamic() && ::ten::is_matrix<T>::value)
T transposed(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::transposed);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return T(std::move(node));
}
/// Transposed static matrix
template <class T>
   requires(T::is_static() && ::ten::is_smatrix<T>::value)
T transposed(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::transposed);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return T(std::move(node));
}

/// Symmetric tensor
template <class T>
   requires(T::is_dynamic())
T symmetric(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::symmetric);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return T(std::move(node));
}
/// Symmetric static tensor
template <class T>
   requires(T::is_static())
T symmetric(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::symmetric);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return T(std::move(node));
}

// TODO Extend hermitian to tensor

/// Hermitian matrix
template <class T>
   requires(T::is_dynamic() && ::ten::is_matrix<T>::value)
T hermitian(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::hermitian);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return T(std::move(node));
}
/// Hermitian static matrix
template <class T>
   requires(T::is_static() && ::ten::is_smatrix<T>::value)
T hermitian(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::hermitian);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return T(std::move(node));
}

/// Lower triangular matrix
template <class T>
   requires(T::is_dynamic() && ::ten::is_matrix<T>::value)
T lower_tr(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::lower_tr);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return T(std::move(node));
}
/// Lower triangular static matrix
template <class T>
   requires(T::is_static() && ::ten::is_smatrix<T>::value)
T lower_tr(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::lower_tr);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return T(std::move(node));
}

/// Upper triangular matrix
template <class T>
   requires(T::is_dynamic() && ::ten::is_matrix<T>::value)
T upper_tr(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::upper_tr);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return T(std::move(node));
}
/// Upper triangular static matrix
template <class T>
   requires(T::is_static() && ::ten::is_smatrix<T>::value)
T upper_tr(const T &t) {
   using node_type = T::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::upper_tr);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return T(std::move(node));
}

////////////////////////////////////////////////////////////////////////////////
// Columns
template <class T, class Shape, storage_order Order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
// requires(Shape::rank() == 2)
class ranked_column final
    : public ten::expr<ranked_column<T, Shape, Order, Storage, Allocator>>,
      public ::ten::tensor_operations<T, Shape, Order, Storage, Allocator>,
      public ::ten::tensor_base {
   static_assert(Shape::rank() == 2,
                 "ranked_column is defined only for matrices.");

 public:
   /*
   template<class To>
      requires std::convertible_to<T, To>
   using casted_type = ranked_column<To, Shape, Order,
      typename Storage::template casted_type<To>,
      typename details::allocator_type<
         typename Storage::template casted_type<To>>::type>;*/

   /// \typedef base_type
   /// Type of the tensor operations.
   using base_type = tensor_operations<T, Shape, Order, Storage, Allocator>;

   /// \typedef node_type
   /// Node type
   using node_type = tensor_node<T, Shape, Order, Storage, Allocator>;

 private:
   size_t _index = 0;
   std::shared_ptr<node_type> _node = nullptr;

 public:
   /// Constructors
   ranked_column(const std::shared_ptr<node_type> &node,
                 const size_t index) noexcept
       : _node(node), _index(index) {}
   ranked_column(std::shared_ptr<node_type> &&node, const size_t index) noexcept
       : _node(std::move(node)), _index(index) {}

   // TODO Assignment from expression

   /// Copy constructor
   ranked_column(const ranked_column &t) { _node = t._node; }
   /// Move constructor
   ranked_column(ranked_column &&t) { _node = std::move(t._node); }

   /// Assignment operator
   ranked_column &operator=(const ranked_column &t) {
      _node = t._node;
      return *this;
   }
   /// Assignment operator
   ranked_column &operator=(ranked_column &&t) {
      _node = std::move(t._node);
      return *this;
   }

   /// Returns the shape
   [[nodiscard]] inline const Shape &shape() const {
      return _node.get()->shape();
   }

   /// Returns the strides
   [[nodiscard]] inline const typename base_type::stride_type &strides() const {
      return _node.get()->strides();
   }

   /// Returns the dynamic size
   [[nodiscard]] inline size_type size() const {
      return _node.get()->shape().dim(0);
   }

   /// Returns the index'th dynamic dimension
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _node.get()->dim(index);
   }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Get the data
   [[nodiscard]] const T *data() const {
      size_t rows = this->shape().dim(0);
      return _node.get()->data() + _index * rows;
   }

   /// Get the data
   [[nodiscard]] T *data() {
      size_t rows = this->shape().dim(0);
      return _node.get()->data() + _index * rows;
   }

   /// Returns the shared ptr to the storage
   [[nodiscard]] std::shared_ptr<Storage> storage() const {
      return _node.get()->storage();
   }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[index + _index * rows];
   }

   /// Overloading the [] operator
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[index + _index * rows];
   }

   /// Asignment from a static expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_static())
   ranked_column &operator=(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Error: Evaluated type must be a tensor.");

      auto value = expr.eval();

      // FIXME maybe static_assert(__shape::template  ==
      // evaluated_type::static_size(),
      //               "Expected equal shape size.");
      size_t rows = _node.get()->shape().dim(0);
      for (size_t idx = 0; idx < rows; idx++) {
         (*_node.get())[idx + _index * rows] = value[idx];
      }
      return *this;
   }

   /// Asignment from a dynamic expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_dynamic())
   ranked_column &operator=(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Evaluated type must be a tensor.");
      auto value = expr.eval();
      // Copy the data
      size_t rows = _node.get()->shape().dim(0);
      for (size_t idx = 0; idx < rows; idx++) {
         (*_node.get())[idx + _index * rows] = value[idx];
      }
      return *this;
   }

   // Asignement from a static vector
   template <StaticVector V> ranked_column &operator=(V &&value) {
      size_t rows = _node.get()->shape().dim(0);
      for (size_t idx = 0; idx < rows; idx++) {
         (*_node.get())[idx + _index * rows] = value[idx];
      }
      return *this;
   }

   // Asgnement from a dynamic vector
   template <DynamicVector V> ranked_column &operator=(V &&value) {
      size_t rows = _node.get()->shape().dim(0);
      for (size_t idx = 0; idx < rows; idx++) {
         (*_node.get())[idx + _index * rows] = value[idx];
      }
      return *this;
   }

   template <class Type, class ShapeType, storage_order StorageOrder,
             class StorageType, class AllocatorType>
   friend std::ostream &
   operator<<(std::ostream &,
              const ranked_column<Type, ShapeType, StorageOrder, StorageType,
                                  AllocatorType> &);
};

// column<T> or column<T, Shape>
template <class T, class Shape = dynamic_shape<2>,
          storage_order Order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(Shape::rank() == 2)
using column = ranked_column<T, Shape, Order, Storage, Allocator>;

// TODO template<class T, size_t Rows, size_t Cols>
// using scolumn = ranked_column<T, ten::shape<Rows, Cols>>;

/// Overload << operator for column
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_column<
            ranked_column<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_column<T, Shape, order, Storage, Allocator> &t) {
   os << "column<" << ::ten::to_string<T>() << "," << t.size() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}

/*
/// Overload << operator for a static vector
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_svector<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "svector<" << ::ten::to_string<T>() << "," << Shape::static_size()
      << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}*/

////////////////////////////////////////////////////////////////////////////////
// Rows
template <class T, class Shape, storage_order Order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
class ranked_row final
    : public ten::expr<ranked_column<T, Shape, Order, Storage, Allocator>>,
      public ::ten::tensor_operations<T, Shape, Order, Storage, Allocator>,
      public ::ten::tensor_base {
   static_assert(Shape::rank() == 2,
                 "ranked_row is defined only for matrices.");

 public:
   /// \typedef base_type
   /// Type of the tensor operations.
   using base_type = tensor_operations<T, Shape, Order, Storage, Allocator>;

   /// \typedef node_type
   /// Node type
   using node_type = tensor_node<T, Shape, Order, Storage, Allocator>;

 private:
   size_t _index = 0;
   std::shared_ptr<node_type> _node = nullptr;

 public:
   /// Constructors
   ranked_row(const std::shared_ptr<node_type> &node,
              const size_t index) noexcept
       : _node(node), _index(index) {}
   ranked_row(std::shared_ptr<node_type> &&node, const size_t index) noexcept
       : _node(std::move(node)), _index(index) {}

   // TODO Assignment from expression

   /// Copy constructor
   // ranked_row(const ranked_row &t) { _node = t._node; }
   /// Move constructor
   // ranked_row(ranked_row &&t) { _node = std::move(t._node); }

   /// Assignment operator
   /*ranked_row &operator=(const ranked_row &t) {
      _node = t._node;
      return *this;
   }*/
   /// Assignment operator
   /*ranked_row &operator=(ranked_row &&t) {
      _node = std::move(t._node);
      return *this;
   }*/

   /// Returns the shape
   [[nodiscard]] inline const Shape &shape() const {
      return _node.get()->shape();
   }

   /// Returns the strides
   [[nodiscard]] inline const typename base_type::stride_type &strides() const {
      return _node.get()->strides();
   }

   /// Returns the dynamic size
   [[nodiscard]] inline size_type size() const {
      return _node.get()->shape().dim(1);
   }

   /// Returns the index'th dynamic dimension
   [[nodiscard]] inline size_type dim(size_type index) const {
      return _node.get()->dim(index);
   }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Get the data
   [[nodiscard]] const T *data() const { return _node.get()->data() + _index; }

   /// Get the data
   [[nodiscard]] T *data() { return _node.get()->data() + _index; }

   /// Returns the shared ptr to the storage
   [[nodiscard]] std::shared_ptr<Storage> storage() const {
      return _node.get()->storage();
   }

   /// Overloading the [] operator
   [[nodiscard]] inline const typename base_type::value_type &
   operator[](size_type index) const noexcept {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[_index + index * rows];
   }

   /// Overloading the [] operator
   [[nodiscard]] inline typename base_type::value_type &
   operator[](size_type index) noexcept {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[_index + index * rows];
   }

   /// Asignment from a static expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_static())
   ranked_row &operator=(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Error: Evaluated type must be a tensor.");

      auto value = expr.eval();

      // FIXME maybe static_assert(__shape::template  ==
      // evaluated_type::static_size(),
      //               "Expected equal shape size.");
      size_t rows = _node.get()->shape().dim(0);
      size_t cols = _node.get()->shape().dim(1);
      for (size_t idx = 0; idx < cols; idx++) {
         (*_node.get())[_index + idx * rows] = value[idx];
      }
      return *this;
   }

   /// Asignment from a dynamic expression
   template <class ExprType>
      requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
               std::remove_cvref_t<ExprType>::output_type::is_dynamic())
   ranked_row &operator=(ExprType &&expr) noexcept {
      using expr_type = std::remove_cvref_t<ExprType>;
      using evaluated_type = typename expr_type::output_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Evaluated type must be a tensor.");
      auto value = expr.eval();
      // Copy the data
      size_t rows = _node.get()->shape().dim(0);
      size_t cols = _node.get()->shape().dim(1);
      for (size_t idx = 0; idx < cols; idx++) {
         (*_node.get())[_index + idx * rows] = value[idx];
      }
      return *this;
   }

   // Asignement from a static vector
   template <StaticVector V> ranked_row &operator=(V &&value) {
      size_t rows = _node.get()->shape().dim(0);
      size_t cols = _node.get()->shape().dim(1);
      for (size_t idx = 0; idx < cols; idx++) {
         (*_node.get())[_index + idx * rows] = value[idx];
      }
      return *this;
   }

   // Asgnement from a dynamic vector
   template <DynamicVector V> ranked_row &operator=(V &&value) {
      size_t rows = _node.get()->shape().dim(0);
      size_t cols = _node.get()->shape().dim(1);
      for (size_t idx = 0; idx < cols; idx++) {
         (*_node.get())[_index + idx * rows] = value[idx];
      }
      return *this;
   }

   template <class Type, class ShapeType, storage_order StorageOrder,
             class StorageType, class AllocatorType>
   friend std::ostream &
   operator<<(std::ostream &, const ranked_row<Type, ShapeType, StorageOrder,
                                               StorageType, AllocatorType> &);
};

// TODO row
// TODO static rows srow

/// Overload << operator for row
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_row<
            ranked_row<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_row<T, Shape, order, Storage, Allocator> &t) {
   os << "row<" << ::ten::to_string<T>() << "," << t.size() << ">\n";
   size_type size = t.size();
   if (size <= 10) {
      os << t[0];
      for (size_type i = 1; i < t.size(); i++) {
         os << " " << t[i];
      }
   } else {
      os << t[0];
      for (size_type i = 0; i < 5; i++) {
         os << " " << t[i];
      }
      os << "\n...";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << " " << t[i];
      }
   }
   return os;
}

/*
/// Overload << operator for a static vector
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_svector<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value)
std::ostream &operator<<(
    std::ostream &os,
    const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "svector<" << ::ten::to_string<T>() << "," << Shape::static_size()
      << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}*/

////////////////////////////////////////////////////////////////////////////////
// Basic functions

// Cast a tensor
template <class To, DynamicTensor T> auto cast(const T &x) {
   using tensor_type = typename T::template casted_type<To>;
   tensor_type r(x.shape());
   for (size_type i = 0; i < x.size(); i++) {
      r[i] = static_cast<To>(x[i]);
   }
   return r;
}

// Cast a static tensor
template <class To, StaticTensor T> auto cast(const T &x) {
   using tensor_type = typename T::template casted_type<To>;
   tensor_type r;
   for (size_type i = 0; i < x.size(); i++) {
      r[i] = static_cast<To>(x[i]);
   }
   return r;
}

// reshape<shape>(x)
template <class Shape, Expr ExprType>
requires(::ten::is_shape<Shape>::value)
auto reshape(ExprType &&expr) {
   static_assert(Shape::is_static(), "Shape must be static.");

   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename details::output_type<expr_type>::type;
   using reshape_type = typename ::ten::details::reshape_result<output_type, Shape>::type;

   return ::ten::unary_expr<
       expr_type, reshape_type, ::ten::functional::static_reshape<Shape>::template func>(expr);
}

// reshape<dims...>(x)
template <size_type... dims, Expr ExprType> auto reshape(ExprType &&expr) {
   using expr_type = typename std::remove_cvref_t<ExprType>;
   using output_type = typename details::output_type<expr_type>::type;
   using shape_type = ::ten::shape<dims...>;
   using reshape_type = typename ::ten::details::reshape_result<output_type, shape_type>::type;

   return ::ten::unary_expr<expr_type, reshape_type, ::ten::functional::dims_static_reshape<
                                           dims...>::template func>(expr);
}


// reshape(x, shape)
template <Expr ExprType, class Shape>
auto reshape(ExprType &&expr, Shape &&dims) {
   using expr_type = typename std::remove_cvref_t<ExprType>;
   using output_type = typename details::output_type<expr_type>::type;
   using shape_type = std::remove_cvref_t<Shape>;
   using reshape_type = typename ::ten::details::reshape_result<output_type, shape_type>::type;
   return ::ten::unary_expr<
       expr_type, reshape_type, ::ten::functional::dynamic_reshape<shape_type>::template func>(expr, dims);
}

// reshape<rank>(x, shape)
template <size_type Rank, Expr ExprType>
auto reshape(ExprType &&expr, std::initializer_list<size_type> &&dims) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename details::output_type<expr_type>::type;
   using shape_type = ::ten::dynamic_shape<Rank>;
   auto Shape = shape_type(std::move(dims));
   return reshape<expr_type, shape_type>(std::forward<expr_type>(expr), std::move(Shape));
}

// flatten(x)
template <Expr ExprType> auto flatten(ExprType expr) {
   using expr_type = std::remove_cvref_t<ExprType>;

   // tensor
   if constexpr (is_tensor<expr_type>::value) {
      using shape_type = expr_type::shape_type;
      if constexpr (shape_type::is_static()) {
         return reshape<::ten::shape<expr_type::static_size()>, expr_type>(
             std::forward<expr_type>(expr));
      }
      if constexpr (shape_type::is_dynamic()) {
         std::initializer_list<size_type> &&dims = {expr.size()};
         return reshape<1>(std::forward<expr_type>(expr), std::move(dims));
      }
   }

   // unar_expr or binary_expr
   if constexpr (is_unary_expr<expr_type>::value ||
                 is_binary_expr<expr_type>::value) {
      using output_type = typename expr_type::evaluated_type;
      using shape_type = typename output_type::shape_type;
      // TODO static or dynamic shape
      return reshape<::ten::shape<shape_type::static_size()>, expr_type>(
          std::forward<expr_type>(expr));
   }
}

// transpose(x)
template <Expr ExprType> auto transpose(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   static_assert(::ten::is_tensor<expr_type>::value || ::ten::is_column<expr_type>::value || ::ten::is_row<expr_type>::value,
      "Currently transpose support only tensor, column and row.");

   // tensor
   if constexpr (is_tensor<expr_type>::value) {
      using shape_type = expr_type::shape_type;

      if constexpr (shape_type::is_static()) {
         using transpose_type = ::ten::details::static_transpose_result<expr_type, shape_type>::type;
         return ::ten::unary_expr<
             expr_type, transpose_type,
             ::ten::functional::static_transpose<shape_type>::template func>(expr);
      }

      if constexpr (shape_type::is_dynamic()) {
         using transpose_result = ::ten::details::transpose_result<expr_type, shape_type>::type;
         return ::ten::unary_expr<
             expr_type, transpose_result,
             ::ten::functional::dynamic_transpose<shape_type>::template func>(expr);
      }
   }

   // TODO unar_expr or binary_expr
   //if constexpr (is_unary_expr<expr_type>::value ||
    //             is_binary_expr<expr_type>::value) {
    //  using output_type = typename expr_type::evaluated_type;
    //  using shape_type = typename output_type::shape_type;

    //  return reshape<::ten::shape<shape_type::staticSize()>, expr_type>(
   //       std::forward<expr_type>(expr));
   //}
}

////////////////////////////////////////////////////////////////////////////////
// Functions for creating a new tensor

// fill<tensor<...>>(value)
template <class T>
   requires(::ten::is_stensor<T>::value &&
            ::ten::is_static_storage<typename T::storage_type>::value)
[[nodiscard]] auto fill(typename T::value_type value) {
   T x;
   for (size_type i = 0; i < x.size(); i++)
      x[i] = value;
   return x;
}

// fill<T, Shape, order,Storage, Allocator>(value)
template <class T, class Shape, storage_order order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(::ten::is_static_storage<Storage>::value &&
            ::ten::is_stensor<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value)
[[nodiscard]] auto fill(T value) {
   return fill<ranked_tensor<T, Shape, order, Storage, Allocator>>(
       value);
}

// fill<tensor<...>>(Shape, value)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto fill(typename T::shape_type &&shape,
                        typename T::value_type value) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   T x(std::forward<shape_type>(shape));
   for (size_type i = 0; i < x.size(); i++) {
      x[i] = value;
   }
   return x;
}

template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims,
                        typename T::value_type value) {
   using shape_type = typename T::shape_type;
   return fill<T>(shape_type(std::move(dims)), value);
}

// fill<T, Shape, order,Storage, Allocator>(Shape, value)
template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(
       ::ten::is_dynamic_tensor<
           ranked_tensor<T, Shape, order, Storage, Allocator>>::value &&
       ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto fill(Shape &&dims, T value) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   return fill<tensor_type>(std::forward<shape>(dims), value);
}

template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(
       ::ten::is_dynamic_tensor<
           ranked_tensor<T, Shape, order, Storage, Allocator>>::value &&
       ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims, T value) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return fill<tensor_type>(shape_type(std::move(dims)), value);
}

// fill<T, rank>(shape, value)
template <class T, size_type rank, storage_order order = default_order>
[[nodiscard]] auto fill(const dynamic_shape<rank> &shape, T value) {
   using shape_type = ::ten::dynamic_shape<rank>;
   return fill<T, shape_type, order>(std::forward<shape_type>(shape), value);
}

template <class T, size_type rank, storage_order order = default_order>
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims, T value) {
   using shape_type = ::ten::dynamic_shape<rank>;
   return fill<T, shape_type, order>(shape_type(std::move(dims)), value);
}

// zeros<tensor<...>>()
template <class T>
   requires(::ten::is_stensor<T>::value &&
            ::ten::is_static_storage<typename T::storage_type>::value)
[[nodiscard]] auto zeros() {
   using value_type = typename T::value_type;
   return fill<T>(value_type(0));
}

// zeros<T, Shape, order,Storage, Allocator>()
template <class T, class Shape, storage_order order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(::ten::is_stensor<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value &&
            ::ten::is_static_storage<Storage>::value)
[[nodiscard]] auto zeros() {
   return zeros<ranked_tensor<T, Shape, order, Storage, Allocator>>();
}

// zeros<tensor<...>>(shape)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto zeros(typename T::shape_type &&shape) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   return fill<T>(std::forward<shape_type>(shape), value_type(0));
}

template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using shape_type = typename T::shape_type;
   return zeros<T>(shape_type(std::move(dims)));
}

// zeros<T, Shape, order,Storage, Allocator>(shape)
template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto zeros(Shape &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   return zeros<tensor_type>(std::forward<Shape>(dims));
}

template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return zeros<tensor_type>(shape_type(std::move(dims)));
}

// zeros<T, __rank>(shape)
template <class T, size_type __rank, storage_order order = default_order>
[[nodiscard]] auto zeros(const dynamic_shape<__rank> &dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return zeros<T, shape_type, order>(std::forward<shape_type>(dims));
}

template <class T, size_type __rank, storage_order order = default_order>
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return zeros<T, shape_type, order>(shape_type(std::move(dims)));
}

// ones<tensor<...>>()
template <class T>
   requires(::ten::is_stensor<T>::value &&
            ::ten::is_static_storage<typename T::storage_type>::value)
[[nodiscard]] auto ones() {
   using value_type = typename T::value_type;
   return fill<T>(value_type(1));
}

// ones<T, Shape, order,Storage, Allocator>()
template <class T, class Shape, storage_order order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(::ten::is_stensor<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value &&
            ::ten::is_static_storage<Storage>::value)
[[nodiscard]] auto ones() {
   return ones<ranked_tensor<T, Shape, order, Storage, Allocator>>();
}

// ones<tensor<...>>(shape)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto ones(typename T::shape_type &&shape) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   return fill<T>(std::forward<shape_type>(shape), value_type(1));
}

template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using shape_type = typename T::shape_type;
   return ones<T>(shape_type(std::move(dims)));
}

// ones<T, Shape, order,Storage, Allocator>(shape)
template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto ones(Shape &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   return ones<tensor_type>(std::forward<shape>(dims));
}

template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return ones<tensor_type>(shape_type(std::move(dims)));
}

// ones<T, __rank>(shape)
template <class T, size_type __rank, storage_order order = default_order>
[[nodiscard]] auto ones(const dynamic_shape<__rank> &shape) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return ones<T, shape_type, order>(std::forward<shape_type>(shape));
}

template <class T, size_type __rank, storage_order order = default_order>
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return ones<T, shape_type, order>(shape_type(std::move(dims)));
}

// range<tensor<...>>(value)
template <class T>
   requires(::ten::is_stensor<T>::value &&
            ::ten::is_static_storage<typename T::storage_type>::value)
[[nodiscard]] auto
range(typename T::value_type value = typename T::value_type(0)) {
   using value_type = typename T::value_type;
   T x;
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}

// range<T, Shape, storage_order,Storage, Allocator>(value)
template <class T, class Shape, storage_order order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(std::is_floating_point_v<T> &&
            ::ten::is_stensor<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value &&
            ::ten::is_static_storage<Storage>::value)
[[nodiscard]] auto range(T value = T(0)) {
   return range<ranked_tensor<T, Shape, order, Storage, Allocator>>(
       value);
}

// range<tensor<...>>(Shape, value)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto
range(typename T::shape_type &&shape,
      typename T::value_type value = typename T::value_type(0)) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   T x(std::forward<shape_type>(shape));
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto
range(std::initializer_list<size_type> &&dims,
      typename T::value_type value = typename T::value_type(0)) {
   using shape_type = typename T::shape_type;
   return range<T>(shape_type(std::move(dims)), value);
}

// range<vector<...>>(size)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value &&
            T::isvector())
[[nodiscard]] auto
range(size_type size,
      typename T::value_type value = typename T::value_type(0)) {
   using shape_type = typename T::shape_type;
   return range<T>(shape_type({size}), value);
}

// range<T, Shape, order,Storage, Allocator>(Shape, value)
template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto range(Shape &&dims, T value = T(0)) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   return range<tensor_type>(std::forward<Shape>(dims), value);
}

template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto range(std::initializer_list<size_type> &&dims,
                         T value = T(0)) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return range<tensor_type>(shape_type(std::move(dims)), value);
}

// range<T, rank>(Shape, value)
template <class T, size_type __rank = 1,
          storage_order order = default_order>
   requires(std::is_floating_point_v<T>)
[[nodiscard]] auto range(const dynamic_shape<__rank> &shape,
                         T value = T(0)) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return range<T, shape_type, order>(std::forward<shape_type>(shape),
                                          value);
}

template <class T, size_type __rank = 1,
          storage_order order = default_order>
   requires(std::is_floating_point_v<T>)
[[nodiscard]] auto range(std::initializer_list<size_type> &&dims,
                         T value = T(0)) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return range<T, shape_type, order>(shape_type(std::move(dims)), value);
}

// linear<stensor<...>>(start, stop)
template <class T>
   requires(::ten::is_stensor<T>::value &&
            ::ten::is_static_storage<typename T::storage_type>::value)
[[nodiscard]] auto linear(typename T::value_type start,
                          typename T::value_type stop) {
   using value_type = typename T::value_type;
   T x;
   x[0] = start;
   constexpr size_t n = T::static_size();
   static_assert(n > 1, "n must be greater than 1.");
   value_type step = (stop - start) / (n - 1);
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + step;
   }
   return x;
}

// linear<T, Shape, storage_order,Storage, Allocator>(start, stop)
template <class T, class Shape, storage_order order = default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
   requires(std::is_floating_point_v<T> &&
            ::ten::is_stensor<ranked_tensor<T, Shape, order, Storage,
                                            Allocator>>::value &&
            ::ten::is_static_storage<Storage>::value)
[[nodiscard]] auto linear(T start, T stop) {
   return linear<ranked_tensor<T, Shape, order, Storage, Allocator>>(
       start, stop);
}

// linear<tensor<...>>(start, stop, Shape)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto linear(typename T::value_type start,
                          typename T::value_type stop,
                          typename T::shape_type &&shape) {
   using value_type = typename T::value_type;
   using shape_type = typename T::shape_type;
   T x(std::forward<shape_type>(shape));
   x[0] = start;
   size_t n = shape.size();
   value_type step = (stop - start) / (n - 1);
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + step;
   }
   return x;
}
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value)
[[nodiscard]] auto linear(typename T::value_type start,
                          typename T::value_type stop,
                          std::initializer_list<size_type> &&dims) {
   using shape_type = typename T::shape_type;
   return linear<T>(start, stop, shape_type(std::move(dims)));
}
/*
// linear<vector<...>>(start, stop, size)
template <class T>
   requires(::ten::is_dynamic_tensor<T>::value &&
            ::ten::is_dense_storage<typename T::storage_type>::value &&
            T::isvector())
[[nodiscard]] auto
linear(typename T::value_type start, typename T::value_type stop, size_type
size){ using shape_type = typename T::shape_type; return linear<T>(start,
stop, shape_type({size}));
}*/

// linear<T, Shape, storage_order,Storage, Allocator>(start, stop,
// Shape)
template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto linear(T start, T stop, Shape &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   return linear<tensor_type>(start, stop, std::forward<Shape>(dims));
}

template <class T, class Shape, storage_order order = default_order,
          class Storage = ::ten::default_storage<T, Shape>,
          class Allocator =
              typename ::ten::details::allocator_type<Storage>::type>
   requires(::ten::is_dynamic_tensor<ranked_tensor<
                T, Shape, order, Storage, Allocator>>::value &&
            ::ten::is_dense_storage<Storage>::value)
[[nodiscard]] auto linear(T start, T stop,
                          std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<T, Shape, order, Storage, Allocator>;
   using shape_type = typename tensor_type::shape_type;
   return linear<tensor_type>(start, stop, shape_type(std::move(dims)));
}

// linear<T, __rank>(start, stop, Shape)
template <class T, size_type __rank = 1,
          storage_order order = default_order>
   requires(std::is_floating_point_v<T>)
[[nodiscard]] auto linear(T start, T stop,
                          const dynamic_shape<__rank> &dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return linear<T, shape_type, order>(start, stop,
                                           std::forward<shape_type>(dims));
}

template <class T, size_type __rank = 1,
          storage_order order = default_order>
   requires(std::is_floating_point_v<T>)
[[nodiscard]] auto linear(T start, T stop,
                          std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return linear<T, shape_type, order>(start, stop,
                                           shape_type(std::move(dims)));
}

////////////////////////////////////////////////////////////////////////////////
// Conversion from special matrices and tensors

/// Convert diagonal matrix to dense
template <Diagonal T> auto dense(T x) -> decltype(auto) {
   using diagonal_type = std::remove_cvref_t<T>;
   using value_type = diagonal_type::value_type;
   size_type m = x.dim(0);
   size_type n = x.dim(1);
   if (m != n) {
      std::cerr << "Diagonal matrix must be square\n";
   }
   matrix<value_type> y = ten::zeros<ten::matrix<value_type>>({m, n});
   for (size_t i = 0; i < m; i++) {
      y(i, i) = x[i];
   }
   return y;
}

/// Convert diagonal matrix to dense
template <SDiagonal T> auto dense(T x) -> decltype(auto) {
   using diagonal_type = std::remove_cvref_t<T>;
   using value_type = diagonal_type::value_type;
   using shape_type = diagonal_type::shape_type;
   constexpr size_type m = shape_type::template static_dim<0>();
   constexpr size_type n = shape_type::template static_dim<1>();
   static_assert(m == n, "Matrix must be square");
   smatrix<value_type, m, n> y = ten::zeros<ten::smatrix<value_type, m, n>>();
   for (size_t i = 0; i < m; i++) {
      y(i, i) = x[i];
   }
   return y;
}

////////////////////////////////////////////////////////////////////////////////
// Expressions

// Gemm
// C <- alpha * X * Y + beta * C
template <Expr X, Expr Y, Tensor C, class T>
   requires(std::is_same_v<T, typename C::value_type>)
void gemm(const T alpha, X &&x, Y &&y, const T beta, C &c) {
   using x_expr_type = std::remove_cvref_t<X>;
   using y_expr_type = std::remove_cvref_t<Y>;

   if constexpr (::ten::is_tensor<x_expr_type>::value &&
                 ::ten::is_tensor<y_expr_type>::value) {
      ::ten::kernels::mul_add(std::forward<X>(x), std::forward<Y>(y), c, alpha,
                              beta);
   }

   if constexpr (::ten::is_tensor<x_expr_type>::value &&
                 !::ten::is_tensor<y_expr_type>::value) {
      auto ytensor = y.eval();
      using YTensor = decltype(ytensor);
      ::ten::kernels::mul_add(std::forward<X>(x),
                              std::forward<YTensor>(ytensor), c, alpha, beta);
   }

   if constexpr (!::ten::is_tensor<x_expr_type>::value &&
                 ::ten::is_tensor<y_expr_type>::value) {
      auto xtensor = x.eval();
      using XTensor = decltype(xtensor);
      ::ten::kernels::mul_add(std::forward<XTensor>(xtensor),
                              std::forward<Y>(y), c, alpha, beta);
   }

   if constexpr (!::ten::is_tensor<x_expr_type>::value &&
                 !::ten::is_tensor<y_expr_type>::value) {
      auto xtensor = x.eval();
      using XTensor = decltype(xtensor);
      auto ytensor = y.eval();
      using YTensor = decltype(ytensor);
      ::ten::kernels::mul_add(std::forward<XTensor>(x),
                              std::forward<YTensor>(y), c, alpha, beta);
   }
}

// axpy
// y <- a*x + y
template <Expr X, Tensor Y, typename T>
   requires(std::is_same_v<T, typename Y::value_type>)
void axpy(const T a, X &&x, Y &y) {
   using x_expr_type = std::remove_cvref_t<X>;

   if constexpr (::ten::is_tensor<x_expr_type>::value) {
      ::ten::kernels::axpy(a, std::forward<X>(x), y);
   }
   if constexpr (!::ten::is_tensor<x_expr_type>::value) {
      auto xtensor = x.eval();
      using XTensor = decltype(xtensor);
      ::ten::kernels::axpy(a, std::forward<XTensor>(xtensor), y);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// functions

/// \fn min
/// Returns the maximum of an expression
template <Expr ExprType> auto min(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   // using output_type = typename ::ten::details::output_type<expr_type>::type;
   using value_type = typename expr_type::value_type;
   using output_type = ten::scalar<value_type>;
   return unary_expr<expr_type, output_type, functional::min>(expr);
}

/// \fn max
/// Return the maximum of an tensor or an expression
template <Expr ExprType> auto max(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using value_type = typename expr_type::value_type;
   using output_type = ten::scalar<value_type>;
   return unary_expr<expr_type, output_type, functional::max>(expr);
}

/// \fn sum
/// Return the sum of a tensor or an expression
template <Expr ExprType> auto sum(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using value_type = typename expr_type::value_type;
   using output_type = ten::scalar<value_type>;
   return unary_expr<expr_type, output_type, functional::sum>(expr);
}

/// \fn cum_sum
/// Return the cumulative sum of a tensor or an expression
template <Expr ExprType> auto cum_sum(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::cum_sum>(expr);
}

/// \fn prod
/// Return the maximum of an tensor or an expression
template <Expr ExprType> auto prod(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using value_type = typename expr_type::value_type;
   using output_type = ten::scalar<value_type>;
   return unary_expr<expr_type, output_type, functional::prod>(expr);
}

/// \fn abs
/// Returns the absolute value of a scalar, a tensor or an expression
template <Expr ExprType> auto abs(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::abs>(expr);
}

/// \fn sqrt
/// Returns the square root of a scalar, a tensor or an expression
template <Expr ExprType> auto sqrt(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::sqrt>(expr);
}

/// \fn sqr
/// Returns the square of an expression
template <Expr ExprType> auto sqr(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::sqr>(expr);
}

/// \fn sin
/// Returns the sine of a scalar, a tensor or an expression
template <Expr ExprType> auto sin(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::sin>(expr);
}

/// \fn sinh
/// Hyperbolic sine
template <Expr ExprType> auto sinh(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::sinh>(expr);
}

/// \fn asin
/// Arc sine
template <Expr ExprType> auto asin(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::asin>(expr);
}

/// \fn cos
/// Returns the cosine of a scalar, a tensor or an expression
template <Expr ExprType> auto cos(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::cos>(expr);
}

/// \fn acos
template <Expr ExprType> auto acos(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::acos>(expr);
}

/// \fn cosh
template <Expr ExprType> auto cosh(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::cosh>(expr);
}

/// \fn tan
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto tan(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::tan>(expr);
}

/// \fn atan
template <Expr ExprType> auto atan(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::atan>(expr);
}

/// \fn tanh
template <Expr ExprType> auto tanh(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::tanh>(expr);
}

/// \fn exp
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto exp(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::exp>(expr);
}

/// \fn log
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto log(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::log>(expr);
}

/// \fn log10
template <Expr ExprType> auto log10(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::log10>(expr);
}

/// \fn floor
template <Expr ExprType> auto floor(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::floor>(expr);
}

/// \fn ceil
template <Expr ExprType> auto ceil(ExprType &&expr) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::ceil>(expr);
}

////////////////////////////////////////////////////////////////////////////////
// Parameteric functions

/// \fn pow
/// Power
template <Expr ExprType, class T = float> auto pow(ExprType &&expr, T n) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using output_type = typename ::ten::details::output_type<expr_type>::type;
   return unary_expr<expr_type, output_type, functional::pow>(expr, n);
}

////////////////////////////////////////////////////////////////////////////////
// Tensor functions

template <Tensor __t> bool all_close(const __t &t, double eps = 1e-5) {
   bool res = true;
   for (size_t i = 0; i < t.size(); i++) {
      if (std::abs(t[i]) >= eps) {
         res = false;
         break;
      }
   }
   return res;
}

} // namespace ten

#endif
