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
template <Expr left_expr, Expr right_expr>
auto operator+(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<right_expr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::binary_func<::ten::binary_operation::add>::func>(
       left.node(), right.node());
}

template <typename T, Expr E>
auto operator+(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return ::ten::scalar<T>(scalar) + std::forward<R>(expr);
}

template <Expr E, typename T>
   requires std::is_floating_point_v<T>
auto operator+(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) + ::ten::scalar<T>(scalar);
}

// Substract two expressions
template <Expr left_expr, Expr right_expr>
auto operator-(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<right_expr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::binary_func<::ten::binary_operation::sub>::func>(
       left.node(), right.node());
}

/*template <typename T, typename E>
   requires ::ten::is_expr<std::remove_cvref_t<E>>
auto operator-(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return ::ten::scalar<T>(scalar) - std::forward<R>(expr);
}*/

template <Expr E, typename T>
auto operator-(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) - ::ten::scalar<T>(scalar);
}

// Multiply two expressions
template <Expr left_expr, Expr right_expr>
auto operator*(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<right_expr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::mul<typename L::node_type,
                              typename R::node_type>::template func>(
       left.node(), right.node());
}

template <typename T, Expr E>
auto operator*(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return ::ten::scalar<T>(scalar) * std::forward<R>(expr);
}

template <Expr E, typename T>
auto operator*(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) * ::ten::scalar<T>(scalar);
}

// Divide two expressions
template <Expr left_expr, Expr right_expr>
auto operator/(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<right_expr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::binary_func<::ten::binary_operation::div>::func>(
       left.node(), right.node());
}

/*template <typename T, typename E>
   requires ::ten::is_expr<std::remove_cvref_t<E>>
auto operator/(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   return scalar<T>(scalar) / std::forward<R>(expr);
}*/

template <Expr E, typename T>
auto operator/(E &&expr, T &&scalar) {
   using R = std::remove_cvref_t<E>;
   return std::forward<R>(expr) / ::ten::scalar<T>(scalar);
}

/// AxpyOne
/// A += B
/*template <typename left_expr, typename right_expr>
   requires ::ten::is_expr<std::remove_cvref_t<left_expr>> &&
            ::ten::is_expr<std::remove_cvref_t<right_expr>>
auto operator+=(left_expr &&left, right_expr &&right) {
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

/// \class scalar_node
/// Node of scalar type
template <typename __t>
class scalar_node : public scalar_operations<scalar_node<__t>> {
 public:
   using scalar_type = scalar<__t>;
   using value_type = __t;
   // for std::conditional_t
   using tensor_type = void;
   using shape_type = shape<1>;

 private:
   __t _value;

 public:
   scalar_node() : _value(__t()) {}

   explicit scalar_node(const __t &value) : _value(value) {}
   explicit scalar_node(__t &&value) : _value(std::move(value)) {}

   const __t &value() const { return _value; }

   scalar_node &operator=(const __t &value) {
      _value = value;
      return *this;
   }
};

/// \class scalar
/// Hold a single value of type __t.
template <typename __t>
class scalar : public expr<scalar<__t>>, public scalar_operations<scalar<__t>> {
 public:
   using node_type = scalar_node<__t>;
   using value_type = __t;

 private:
   std::shared_ptr<node_type> _node = nullptr;

 public:
   explicit scalar(const __t &value)
       : _node(std::make_shared<node_type>(value)) {}

   explicit scalar(__t &&value)
       : _node(std::make_shared<node_type>(std::move(value))) {}

   explicit scalar(std::shared_ptr<node_type> node) : _node(node) {}

   /// Asignment from an expression
   template <class Expr>
      requires(::ten::is_unary_expr<std::remove_cvref_t<Expr>>::value ||
               ::ten::is_binary_expr<std::remove_cvref_t<Expr>>::value)
   scalar(Expr &&expr) {
      _node = expr.eval().node();
   }

   /// Get the value
   const __t &value() const { return _node.get()->value(); }

   // Returns the shared ptr to the node
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   // ostream operator
   template <class __r>
   friend std::ostream &operator<<(std::ostream &, const scalar<__r> &);
};

template <class __t>
std::ostream &operator<<(std::ostream &os, const scalar<__t> &s) {
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
   std::shared_ptr<__storage> _storage = nullptr;

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
   explicit tensor_node(const std::shared_ptr<__storage> &st) noexcept
      requires(__shape::is_static())
       : _storage(st) {}

   /// Construct a tensor_node from storage and shape
   /// FIXME Check size
   explicit tensor_node(const std::shared_ptr<__storage> &st,
                        const __shape &dims) noexcept
      requires(__shape::is_dynamic())
       : _storage(st), _shape(dims),
         _stride(typename base_type::stride_type(_shape.value())) {}

   /// Construct a tenso_node from storage and format
   explicit tensor_node(const std::shared_ptr<__storage> &st,
                        ::ten::storage_format format) noexcept
      requires(__shape::is_static())
       : _format(format), _storage(st) {}

   /// Construct a tenso_node from storage and shape
   /// FIXME Check size
   explicit tensor_node(const std::shared_ptr<__storage> &st,
                        const __shape &dims,
                        ::ten::storage_format format) noexcept
      requires(__shape::is_dynamic())
       : _format(format), _storage(st), _shape(dims),
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
   [[nodiscard]] std::shared_ptr<__storage> storage() const { return _storage; }

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
};

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

   /// Asignment from a static expression
   template <class __expr>
      requires((::ten::is_unary_expr<std::remove_cvref_t<__expr>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<__expr>>::value) &&
               std::remove_cvref_t<__expr>::evaluated_type::is_static())
   ranked_tensor(__expr &&expr) noexcept {
      using expr_type = std::remove_cvref_t<__expr>;
      using evaluated_type = typename expr_type::evaluated_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Error: Evaluated type must be a tensor.");

      auto node = expr.eval().node();
      auto st = node.get()->storage();

      static_assert(__shape::static_size() == evaluated_type::static_size() &&
                    "Expected equal shape size.");
      _node = std::make_shared<node_type>(st);
   }

   /// Asignment from a dynamic expression
   template <class __expr>
      requires((::ten::is_unary_expr<std::remove_cvref_t<__expr>>::value ||
                ::ten::is_binary_expr<std::remove_cvref_t<__expr>>::value) &&
               std::remove_cvref_t<__expr>::evaluated_type::is_dynamic())
   ranked_tensor(__expr &&expr) noexcept {
      using expr_type = std::remove_cvref_t<__expr>;
      using evaluated_type = typename expr_type::evaluated_type;

      static_assert(::ten::is_tensor<evaluated_type>::value,
                    "Evaluated type must be a tensor.");

      auto node = expr.eval().node();
      auto st = node.get()->storage();
      auto dims = node.get()->shape();
      _node = std::make_shared<node_type>(st, dims);
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
   /*{
      if constexpr (base_type::shape_type::is_dynamic()) {
         _node = std::make_shared<node_type>(t.storage(), t.shape());
      } else {
         _node = std::make_shared<node_type>(t.storage());
      }
      //_node = t._node;
      return *this;
   }*/

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

   /// Returns the shared ptr to the storage
   [[nodiscard]] std::shared_ptr<__storage> storage() const {
      return _node.get()->storage();
   }

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

   // copy
   auto copy() const {
      auto format = _node.get()->format();
      if constexpr (__shape::is_dynamic()) {
         auto shape = _node.get()->shape();
         auto st = std::make_shared<typename node_type::storage_type>(shape);
         auto node = std::make_shared<node_type>(std::move(st), shape, format);
         ranked_tensor t(std::move(node));
         for (size_t i = 0; i < this->size(); i++) {
            t[i] = (*_node.get())[i];
         }
         return t;
      } else {
         auto st = std::make_shared<typename node_type::storage_type>();
         auto node = std::make_shared<node_type>(std::move(st), format);
         ranked_tensor t(std::move(node));
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

   /// std::ostream friend function
   template <class __type, class __shape_type, storage_order __storage_order,
             class __storage_type, class __allocator_type>
   friend std::ostream &
   operator<<(std::ostream &os,
              const ranked_tensor<__type, __shape_type, __storage_order,
                                  __storage_type, __allocator_type> &t);
};

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

/// FIXME Print only some values

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

// Save vector to a file
template <class T>
   requires(::ten::is_vector<T>::value &&
            std::is_floating_point_v<typename T::value_type>)
void save(const T &t, std::string filename) {
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
void save(const T &t, std::string filename) {
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
      std::cout << "Unsuported file extension" << std::endl;
      return;
   }
   size_t m = t.dim(0);
   size_t n = t.dim(1);

   std::ofstream file(filename, std::ios::out);
   file << "%%MatrixMarket matrix array real general\n";
   file << "%\n";
   file << m << " " << n << "\n";

   for (size_t i = 0; i < m; i++) {
      file << t(i, 0);
      for (size_t j = 1; j < n; j++) {
         file << " " << t(i, j);
      }
      file << "\n";
   }
}

// vector<__t>
template <class __t, storage_order __order = default_order,
          class __storage = default_storage<__t, shape<::ten::dynamic>>,
          class __allocator = typename details::allocator_type<__storage>::type>
using vector =
    ranked_tensor<__t, shape<::ten::dynamic>, __order, __storage, __allocator>;

// svector<__t, size>
template <class __t, size_type __size, storage_order __order = default_order,
          class __storage = default_storage<__t, shape<__size>>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(__size > 0)
using svector =
    ranked_tensor<__t, shape<__size>, __order, __storage, __allocator>;

// matrix<__t> or matrix<__t, __shape>
template <class __t, class __shape = dynamic_shape<2>,
          storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(__shape::rank() == 2)
using matrix = ranked_tensor<__t, __shape, __order, __storage, __allocator>;

// smatrix<__t, rows, cols>
template <class __t, size_type __rows, size_type __cols,
          storage_order __order = default_order,
          class __storage = default_storage<__t, shape<__rows, __cols>>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(__rows > 0 && __cols > 0)
using smatrix =
    ranked_tensor<__t, shape<__rows, __cols>, __order, __storage, __allocator>;

/// \typedef tensor
/// tensor<__t, __rank>
template <class __t, size_type __rank, storage_order __order = default_order,
          class __storage = default_storage<__t, dynamic_shape<__rank>>,
          class __allocator = typename details::allocator_type<__storage>::type>
using tensor =
    ranked_tensor<__t, dynamic_shape<__rank>, __order, __storage, __allocator>;

/// \typedef stensor
/// stensor<__t, __dims...>
template <class __t, size_type... __dims>
// TODO Shorter requires all dims > 0 without shape
   requires(shape<__dims...>::is_static())
using stensor =
    ranked_tensor<__t, shape<__dims...>, default_order,
                  default_storage<__t, shape<__dims...>>,
                  typename details::allocator_type<
                      default_storage<__t, shape<__dims...>>>::type>;

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

/// __transposed matrix
template <class __t>
   requires(__t::is_dynamic() && ::ten::is_matrix<__t>::value)
__t transposed(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::transposed);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return __t(std::move(node));
}
/// __transposed static matrix
template <class __t>
   requires(__t::is_static() && ::ten::is_smatrix<__t>::value)
__t transposed(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::transposed);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return __t(std::move(node));
}

/// Symmetric tensor
template <class __t>
   requires(__t::is_dynamic())
__t symmetric(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::symmetric);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return __t(std::move(node));
}
/// Symmetric static tensor
template <class __t>
   requires(__t::is_static())
__t symmetric(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::symmetric);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return __t(std::move(node));
}

// TODO Extend hermitian to tensor

/// Hermitian matrix
template <class __t>
   requires(__t::is_dynamic() && ::ten::is_matrix<__t>::value)
__t hermitian(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::hermitian);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return __t(std::move(node));
}
/// Hermitian static matrix
template <class __t>
   requires(__t::is_static() && ::ten::is_smatrix<__t>::value)
__t hermitian(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::hermitian);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return __t(std::move(node));
}

/// Lower triangular matrix
template <class __t>
   requires(__t::is_dynamic() && ::ten::is_matrix<__t>::value)
__t lower_tr(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::lower_tr);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return __t(std::move(node));
}
/// Lower triangular static matrix
template <class __t>
   requires(__t::is_static() && ::ten::is_smatrix<__t>::value)
__t lower_tr(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::lower_tr);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return __t(std::move(node));
}

/// Upper triangular matrix
template <class __t>
   requires(__t::is_dynamic() && ::ten::is_matrix<__t>::value)
__t upper_tr(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   auto dims = t.shape();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::upper_tr);
   std::shared_ptr<node_type> node =
       std::make_shared<node_type>(st, dims, format);
   return __t(std::move(node));
}
/// Upper triangular static matrix
template <class __t>
   requires(__t::is_static() && ::ten::is_smatrix<__t>::value)
__t upper_tr(const __t &t) {
   using node_type = __t::node_type;
   auto st = t.storage();
   ::ten::storage_format format = static_cast<storage_format>(
       t.format() | ::ten::storage_format::upper_tr);
   std::shared_ptr<node_type> node = std::make_shared<node_type>(st, format);
   return __t(std::move(node));
}

////////////////////////////////////////////////////////////////////////////////
// Basic functions

// Cast a tensor
template <class __to, DynamicTensor __t>
auto cast(const __t &x) {
   using tensor_type = typename __t::template casted_type<__to>;
   tensor_type r(x.shape());
   for (size_type i = 0; i < x.size(); i++) {
      r[i] = static_cast<__to>(x[i]);
   }
   return r;
}

// Cast a static tensor
template <class __to, StaticTensor __t>
auto cast(const __t &x) {
   using tensor_type = typename __t::template casted_type<__to>;
   tensor_type r;
   for (size_type i = 0; i < x.size(); i++) {
      r[i] = static_cast<__to>(x[i]);
   }
   return r;
}

// reshape<shape>(x)
template <class __shape, Expr __expr>
auto reshape(__expr &&expr) {
   using node_type = typename std::remove_cvref_t<__expr>::node_type;
   return ::ten::unary_expr<
       node_type, ::ten::functional::static_reshape<__shape>::template func>(
       expr.node());
}

// reshape<dims...>(x)
template <size_type... dims, Expr __expr>
auto reshape(__expr &&expr) {
   using node_type = typename std::remove_cvref_t<__expr>::node_type;
   return ::ten::unary_expr<node_type, ::ten::functional::dims_static_reshape<
                                           dims...>::template func>(
       expr.node());
}

// reshape(x, shape)
template <Expr __expr, class __shape>
auto reshape(__expr &&expr, __shape &&dims) {
   using node_type = typename std::remove_cvref_t<__expr>::node_type;
   using shape_type = std::remove_cvref_t<__shape>;
   return ::ten::unary_expr<
       node_type, ::ten::functional::dynamic_reshape<shape_type>::template func,
       shape_type>(expr.node(), dims);
}

// reshape<rank>(x, shape)
template <size_type __rank, Expr __expr>
auto reshape(__expr &&expr, std::initializer_list<size_type> &&dims) {
   using expr_type = std::remove_cvref_t<__expr>;
   using shape_type = ::ten::dynamic_shape<__rank>;
   return reshape<expr_type, shape_type>(std::forward<expr_type>(expr),
                                         shape_type(std::move(dims)));
}

// flatten(x)
template <Expr __expr>
auto flatten(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;

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
template <Expr __expr>
auto transpose(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;

   // tensor
   if constexpr (is_tensor<expr_type>::value) {
      using shape_type = expr_type::shape_type;
      using node_type = expr_type::node_type;

      if constexpr (shape_type::is_static()) {
         return ::ten::unary_expr<
             typename expr_type::node_type,
             ::ten::functional::static_transpose<shape_type>::template func,
             shape_type>(expr.node());
      }

      if constexpr (shape_type::is_dynamic()) {
         return ::ten::unary_expr<
             node_type,
             ::ten::functional::dynamic_transpose<shape_type>::template func,
             shape_type>(expr.node());
      }
   }

   /* TODO unar_expr or binary_expr
   if constexpr (is_unary_expr<expr_type>::value ||
                 is_binary_expr<expr_type>::value) {
      using output_type = typename expr_type::evaluated_type;
      using shape_type = typename output_type::shape_type;

      return reshape<::ten::shape<shape_type::staticSize()>, expr_type>(
          std::forward<expr_type>(expr));
   }*/
}

////////////////////////////////////////////////////////////////////////////////
// Functions for creating a new tensor

// fill<tensor<...>>(value)
template <class __t>
   requires(::ten::is_stensor<__t>::value &&
            ::ten::is_static_storage<typename __t::storage_type>::value)
[[nodiscard]] auto fill(typename __t::value_type value) {
   __t x;
   for (size_type i = 0; i < x.size(); i++)
      x[i] = value;
   return x;
}

// fill<__t, __shape, __order,__storage, __allocator>(value)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(::ten::is_static_storage<__storage>::value &&
            ::ten::is_stensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value)
[[nodiscard]] auto fill(__t value) {
   return fill<ranked_tensor<__t, __shape, __order, __storage, __allocator>>(
       value);
}

// fill<tensor<...>>(__shape, value)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto fill(typename __t::shape_type &&shape,
                        typename __t::value_type value) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   __t x(std::forward<shape_type>(shape));
   for (size_type i = 0; i < x.size(); i++) {
      x[i] = value;
   }
   return x;
}

template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims,
                        typename __t::value_type value) {
   using shape_type = typename __t::shape_type;
   return fill<__t>(shape_type(std::move(dims)), value);
}

// fill<__t, __shape, __order,__storage, __allocator>(__shape, value)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto fill(__shape &&dims, __t value) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   return fill<tensor_type>(std::forward<shape>(dims), value);
}

template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims, __t value) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   using shape_type = typename tensor_type::shape_type;
   return fill<tensor_type>(shape_type(std::move(dims)), value);
}

// fill<__t, rank>(__shape, value)
template <class __t, size_type rank, storage_order __order = default_order>
[[nodiscard]] auto fill(const dynamic_shape<rank> &shape, __t value) {
   using shape_type = ::ten::dynamic_shape<rank>;
   return fill<__t, shape_type, __order>(std::forward<shape_type>(shape),
                                         value);
}

template <class __t, size_type rank, storage_order __order = default_order>
[[nodiscard]] auto fill(std::initializer_list<size_type> &&dims, __t value) {
   using shape_type = ::ten::dynamic_shape<rank>;
   return fill<__t, shape_type, __order>(shape_type(std::move(dims)), value);
}

// zeros<tensor<...>>()
template <class __t>
   requires(::ten::is_stensor<__t>::value &&
            ::ten::is_static_storage<typename __t::storage_type>::value)
[[nodiscard]] auto zeros() {
   using value_type = typename __t::value_type;
   return fill<__t>(value_type(0));
}

// zeros<__t, __shape, __order,__storage, __allocator>()
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(::ten::is_stensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_static_storage<__storage>::value)
[[nodiscard]] auto zeros() {
   return zeros<ranked_tensor<__t, __shape, __order, __storage, __allocator>>();
}

// zeros<tensor<...>>(shape)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto zeros(typename __t::shape_type &&shape) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   return fill<__t>(std::forward<shape_type>(shape), value_type(0));
}

template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using shape_type = typename __t::shape_type;
   return zeros<__t>(shape_type(std::move(dims)));
}

// zeros<__t, __shape, __order,__storage, __allocator>(shape)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto zeros(__shape &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   return zeros<tensor_type>(std::forward<__shape>(dims));
}

template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   using shape_type = typename tensor_type::shape_type;
   return zeros<tensor_type>(shape_type(std::move(dims)));
}

// zeros<__t, __rank>(shape)
template <class __t, size_type __rank, storage_order __order = default_order>
[[nodiscard]] auto zeros(const dynamic_shape<__rank> &dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return zeros<__t, shape_type, __order>(std::forward<shape_type>(dims));
}

template <class __t, size_type __rank, storage_order __order = default_order>
[[nodiscard]] auto zeros(std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return zeros<__t, shape_type, __order>(shape_type(std::move(dims)));
}

// ones<tensor<...>>()
template <class __t>
   requires(::ten::is_stensor<__t>::value &&
            ::ten::is_static_storage<typename __t::storage_type>::value)
[[nodiscard]] auto ones() {
   using value_type = typename __t::value_type;
   return fill<__t>(value_type(1));
}

// ones<__t, __shape, __order,__storage, __allocator>()
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(::ten::is_stensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_static_storage<__storage>::value)
[[nodiscard]] auto ones() {
   return ones<ranked_tensor<__t, __shape, __order, __storage, __allocator>>();
}

// ones<tensor<...>>(shape)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto ones(typename __t::shape_type &&shape) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   return fill<__t>(std::forward<shape_type>(shape), value_type(1));
}

template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using shape_type = typename __t::shape_type;
   return ones<__t>(shape_type(std::move(dims)));
}

// ones<__t, __shape, __order,__storage, __allocator>(shape)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto ones(__shape &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   return ones<tensor_type>(std::forward<shape>(dims));
}

template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   using shape_type = typename tensor_type::shape_type;
   return ones<tensor_type>(shape_type(std::move(dims)));
}

// ones<__t, __rank>(shape)
template <class __t, size_type __rank, storage_order __order = default_order>
[[nodiscard]] auto ones(const dynamic_shape<__rank> &shape) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return ones<__t, shape_type, __order>(std::forward<shape_type>(shape));
}

template <class __t, size_type __rank, storage_order __order = default_order>
[[nodiscard]] auto ones(std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return ones<__t, shape_type, __order>(shape_type(std::move(dims)));
}

// range<tensor<...>>(value)
template <class __t>
   requires(::ten::is_stensor<__t>::value &&
            ::ten::is_static_storage<typename __t::storage_type>::value)
[[nodiscard]] auto
range(typename __t::value_type value = typename __t::value_type(0)) {
   using value_type = typename __t::value_type;
   __t x;
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}

// range<__t, __shape, storage_order,__storage, __allocator>(value)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(std::is_floating_point_v<__t> &&
            ::ten::is_stensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_static_storage<__storage>::value)
[[nodiscard]] auto range(__t value = __t(0)) {
   return range<ranked_tensor<__t, __shape, __order, __storage, __allocator>>(
       value);
}

// range<tensor<...>>(__shape, value)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto
range(typename __t::shape_type &&shape,
      typename __t::value_type value = typename __t::value_type(0)) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   __t x(std::forward<shape_type>(shape));
   x[0] = value;
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto
range(std::initializer_list<size_type> &&dims,
      typename __t::value_type value = typename __t::value_type(0)) {
   using shape_type = typename __t::shape_type;
   return range<__t>(shape_type(std::move(dims)), value);
}

// range<vector<...>>(size)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value &&
            __t::isvector())
[[nodiscard]] auto
range(size_type size,
      typename __t::value_type value = typename __t::value_type(0)) {
   using shape_type = typename __t::shape_type;
   return range<__t>(shape_type({size}), value);
}

// range<__t, __shape, __order,__storage, __allocator>(__shape, value)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto range(__shape &&dims, __t value = __t(0)) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   return range<tensor_type>(std::forward<shape>(dims), value);
}

template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto range(std::initializer_list<size_type> &&dims,
                         __t value = __t(0)) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   using shape_type = typename tensor_type::shape_type;
   return range<tensor_type>(shape_type(std::move(dims)), value);
}

// range<__t, rank>(__shape, value)
template <class __t, size_type __rank = 1,
          storage_order __order = default_order>
   requires(std::is_floating_point_v<__t>)
[[nodiscard]] auto range(const dynamic_shape<__rank> &shape,
                         __t value = __t(0)) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return range<__t, shape_type, __order>(std::forward<shape_type>(shape),
                                          value);
}

template <class __t, size_type __rank = 1,
          storage_order __order = default_order>
   requires(std::is_floating_point_v<__t>)
[[nodiscard]] auto range(std::initializer_list<size_type> &&dims,
                         __t value = __t(0)) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return range<__t, shape_type, __order>(shape_type(std::move(dims)), value);
}

// linear<stensor<...>>(start, stop)
template <class __t>
   requires(::ten::is_stensor<__t>::value &&
            ::ten::is_static_storage<typename __t::storage_type>::value)
[[nodiscard]] auto linear(typename __t::value_type start,
                          typename __t::value_type stop) {
   using value_type = typename __t::value_type;
   __t x;
   x[0] = start;
   constexpr size_t n = __t::static_size();
   static_assert(n > 1, "n must be greater than 1.");
   value_type step = (stop - start) / (n - 1);
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + step;
   }
   return x;
}

// linear<__t, __shape, storage_order,__storage, __allocator>(start, stop)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
   requires(std::is_floating_point_v<__t> &&
            ::ten::is_stensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_static_storage<__storage>::value)
[[nodiscard]] auto linear(__t start, __t stop) {
   return linear<ranked_tensor<__t, __shape, __order, __storage, __allocator>>(
       start, stop);
}

// linear<tensor<...>>(start, stop, __shape)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto linear(typename __t::value_type start,
                          typename __t::value_type stop,
                          typename __t::shape_type &&shape) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   __t x(std::forward<shape_type>(shape));
   x[0] = start;
   size_t n = shape.size();
   value_type step = (stop - start) / (n - 1);
   for (size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + step;
   }
   return x;
}
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto linear(typename __t::value_type start,
                          typename __t::value_type stop,
                          std::initializer_list<size_type> &&dims) {
   using shape_type = typename __t::shape_type;
   return linear<__t>(start, stop, shape_type(std::move(dims)));
}
/*
// linear<vector<...>>(start, stop, size)
template <class __t>
   requires(::ten::is_dtensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value &&
            __t::isvector())
[[nodiscard]] auto
linear(typename __t::value_type start, typename __t::value_type stop, size_type
size){ using shape_type = typename __t::shape_type; return linear<__t>(start,
stop, shape_type({size}));
}*/

// linear<__t, __shape, storage_order,__storage, __allocator>(start, stop,
// __shape)
template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto linear(__t start, __t stop, __shape &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   return linear<tensor_type>(start, stop, std::forward<shape>(dims));
}

template <class __t, class __shape, storage_order __order = default_order,
          class __storage = ::ten::default_storage<__t, __shape>,
          class __allocator =
              typename ::ten::details::allocator_type<__storage>::type>
   requires(::ten::is_dtensor<ranked_tensor<__t, __shape, __order, __storage,
                                            __allocator>>::value &&
            ::ten::is_dense_storage<__storage>::value)
[[nodiscard]] auto linear(__t start, __t stop,
                          std::initializer_list<size_type> &&dims) {
   using tensor_type =
       ranked_tensor<__t, __shape, __order, __storage, __allocator>;
   using shape_type = typename tensor_type::shape_type;
   return linear<tensor_type>(start, stop, shape_type(std::move(dims)));
}

// linear<__t, __rank>(start, stop, __shape)
template <class __t, size_type __rank = 1,
          storage_order __order = default_order>
   requires(std::is_floating_point_v<__t>)
[[nodiscard]] auto linear(__t start, __t stop,
                          const dynamic_shape<__rank> &dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return linear<__t, shape_type, __order>(start, stop,
                                           std::forward<shape_type>(dims));
}

template <class __t, size_type __rank = 1,
          storage_order __order = default_order>
   requires(std::is_floating_point_v<__t>)
[[nodiscard]] auto linear(__t start, __t stop,
                          std::initializer_list<size_type> &&dims) {
   using shape_type = ::ten::dynamic_shape<__rank>;
   return linear<__t, shape_type, __order>(start, stop,
                                           shape_type(std::move(dims)));
}

////////////////////////////////////////////////////////////////////////////////
// Conversion from special matrices and tensors

/// Convert diagonal matrix to dense
template <Diagonal T> auto dense(T &&x) -> decltype(auto) {
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
template <SDiagonal T> auto dense(T &&x) -> decltype(auto) {
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
/// functions

/// \fn min
/// Returns the maximum of an expression
template <Expr __expr>
auto min(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::min>(
       expr.node());
}

/// \fn max
/// Return the maximum of an tensor or an expression
template <Expr __expr>
auto max(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::max>(
       expr.node());
}

/// \fn sum
/// Return the maximum of an tensor or an expression
template <Expr __expr>
auto sum(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::sum>(
       expr.node());
}

/// \fn cum_sum
/// Return the maximum of an tensor or an expression
template <Expr __expr>
auto cum_sum(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::cum_sum>(
       expr.node());
}
/// \fn sum
/// Return the maximum of an tensor or an expression
template <Expr __expr>
auto prod(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::prod>(
       expr.node());
}

/// \fn abs
/// Returns the absolute value of a scalar, a tensor or an expression
template <Expr __expr>
auto abs(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::abs>(
       expr.node());
}

/// \fn sqrt
/// Returns the square root of a scalar, a tensor or an expression
template <Expr __expr>
auto sqrt(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::sqrt>(
       expr.node());
}

/// \fn sqr
/// Returns the square of an expression
template <Expr __expr>
auto sqr(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::sqr>(
       expr.node());
}

/// \fn sin
/// Returns the sine of a scalar, a tensor or an expression
template <Expr __expr>
auto sin(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::sin>(
       expr.node());
}

/// \fn sinh
/// Hyperbolic sine
template <Expr __expr>
auto sinh(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::sinh>(
       expr.node());
}

/// \fn asin
/// Arc sine
template <Expr __expr>
auto asin(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::asin>(
       expr.node());
}

/// \fn cos
/// Returns the cosine of a scalar, a tensor or an expression
template <Expr __expr>
auto cos(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::cos>(
       expr.node());
}

/// \fn acos
template <Expr __expr>
auto acos(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::acos>(
       expr.node());
}

/// \fn cosh
template <Expr __expr>
auto cosh(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::cosh>(
       expr.node());
}

/// \fn tan
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr __expr>
auto tan(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::tan>(
       expr.node());
}

/// \fn atan
template <Expr __expr>
auto atan(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::atan>(
       expr.node());
}

/// \fn tanh
template <Expr __expr>
auto tanh(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::tanh>(
       expr.node());
}

/// \fn exp
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr __expr>
auto exp(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::exp>(
       expr.node());
}

/// \fn log
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr __expr>
auto log(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   using node_type = expr_type::node_type;
   return unary_expr<typename expr_type::node_type, functional::log>(
       expr.node());
}

/// \fn log10
template <Expr __expr>
auto log10(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::log10>(
       expr.node());
}

/// \fn floor
template <Expr __expr>
auto floor(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::floor>(
       expr.node());
}

/// \fn ceil
template <Expr __expr>
auto ceil(__expr &&expr) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::ceil>(
       expr.node());
}

////////////////////////////////////////////////////////////////////////////////
// Parameteric functions

/// \fn pow
/// Power
template <Expr __expr, class T = float>
auto pow(__expr &&expr, T n) {
   using expr_type = std::remove_cvref_t<__expr>;
   return unary_expr<typename expr_type::node_type, functional::pow>(
       expr.node(), n);
}

////////////////////////////////////////////////////////////////////////////////
// Tensor functions

template <Tensor __t>
bool all_close(const __t &t, double eps = 1e-5) {
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
