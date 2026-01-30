#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <ten/functions.hxx>
#include <ten/types.hxx>

#include <ten/functional.hxx>

#include <ten/autograd.hxx>

namespace ten::details {

// Input type
template <class> struct input_type;

template <class T> struct input_type<scalar<T>> {
   using type = scalar<T>;
};

template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
struct input_type<ranked_tensor<T, Shape, order, Storage, Allocator>> {
   using type = ranked_tensor<T, Shape, order, Storage, Allocator>;
};

template <class Input, class Output, template <typename...> class F,
          class... Args>
struct input_type<::ten::unary_expr<Input, Output, F, Args...>> {
   using type =
       typename ::ten::unary_expr<Input, Output, F, Args...>::input_type;
};

template <class Left, class Right, class Output, template <typename...> class F,
          class... Args>
struct input_type<::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::output_type;
};

// Output type
template <class> struct output_type;

template <class T> struct output_type<scalar<T>> {
   using type = scalar<T>;
};

template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
struct output_type<ranked_tensor<T, Shape, order, Storage, Allocator>> {
   using type = ranked_tensor<T, Shape, order, Storage, Allocator>;
};

template <class Input, class Output, template <typename...> class F,
          class... Args>
struct output_type<::ten::unary_expr<Input, Output, F, Args...>> {
   using type =
       typename ::ten::unary_expr<Input, Output, F, Args...>::output_type;
};

template <class Left, class Right, class Output, template <typename...> class F,
          class... Args>
struct output_type<::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::output_type;
};

// Input shape
template <Tensor T> static inline auto input_shape(T &t) -> decltype(auto) {
   return t.shape();
}

template <UnaryExpr ExprType>
static inline auto input_shape(ExprType &expr) -> decltype(auto) {
   return expr.value().shape();
}

template <BinaryExpr ExprType>
static inline auto input_shape(ExprType &expr) -> decltype(auto) {
   return expr.value().shape();
}

// Input value
template <Scalar T> static inline auto input_value(T &t) { return t; }

template <Tensor T> static inline auto input_value(T &t) { return t; }

template <UnaryExpr ExprType> static inline auto input_value(ExprType &expr) {
   return expr.value();
}

template <BinaryExpr ExprType> static inline auto input_value(ExprType &expr) {
   return expr.value();
}

} // namespace ten::details

namespace ten {

// Unary node
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
struct unary_node {
   using input_type = typename ::ten::details::output_type<Input>::type;
   using output_type = Output;
   using func_type = Func<input_type, output_type>;

   bool _evaluated = false;
   bool _retain_grad = false;
   std::optional<func_type> _func = std::nullopt;
   std::shared_ptr<Input> _input = nullptr;
   std::shared_ptr<Output> _value = nullptr;

   /// Return the input
   [[nodiscard]] inline Input &input() const { return *_input.get(); }

   /// Return the input node
   [[nodiscard]] inline std::shared_ptr<Input> input_node() { return _input; }

   /// Return the output
   [[nodiscard]] inline Output &value() const { return *_value.get(); }

   /// Return the std::shared_ptr to the output
   [[nodiscard]] inline std::shared_ptr<Output> value_node() { return _value; }

   unary_node(Input &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _input(std::make_shared<Input>(inp)), _func(func_type()) {}

   /// Construct a unary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_node(Input &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _input(std::make_shared<Input>(inp)),
         _func(func_type(std::forward<func_args>(fargs)...)) {}

   ~unary_node() {}
};

// \class unary_expr
// Apply a function to a tensor, column or row
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class unary_expr : ten::expr<unary_expr<Input, Output, Func, Args...>> {
 public:
   using input_ty = Input;
   using output_ty = Output;

   using input_type = typename ::ten::details::output_type<Input>::type;
   using output_type = Output;

   using func_type = Func<input_type, output_type>;

   using node_type = unary_node<Input, Output, Func, Args...>;

   using value_type = output_type::value_type;

   using expr_type = unary_expr<Input, Output, Func, Args...>;

 private:
   std::shared_ptr<node_type> _node = nullptr;

 public:
   unary_expr() {}

   ~unary_expr() {}

   /// Construct a unary expr if the function doesn't take additional parameters
   unary_expr(Input &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
   {
      _node = std::make_shared<node_type>(inp);
   }

   /// Construct a unary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_expr(Input &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
   {
      _node =
          std::make_shared<node_type>(inp, std::forward<func_args>(fargs)...);
   }

   /// Requires gradient
   [[nodiscard]] inline bool requires_grad() {
      return _node->_input.get()->requires_grad();
   }

   /// Retain the gradient
   [[nodiscard]] inline bool has_retain_grad() { return _node->_retain_grad; }

   // Retain the gradient for this node
   void retain_grad() { _node->_retain_grad = true; }

   /// Return the input
   [[nodiscard]] inline Input &input() { return *_node->_input.get(); }

   /// Return the input node
   [[nodiscard]] inline std::shared_ptr<Input> input_node() {
      return _node->_input;
   }

   /// Returns whether the expression is evaluated
   [[nodiscard]] inline bool evaluated() { return _node->_evaluated; }

   /// Returns the evaluated expression of type ten::scalar or ten::tensor
   [[nodiscard]] inline output_type &value() { return *_node->_value.get(); }

   /// Return the node to the evaluated expression of type ten::scalar or
   /// ten::tensor
   [[nodiscard]] inline std::shared_ptr<Output> value_node() {
      return _node->_value;
   }

   [[nodiscard]] auto grad() noexcept -> output_type {
      return _node->_value.get()->grad();
   }

   /// Evaluate the expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (evaluated())
         return _node->value();

      // Evaluate input
      using input_type = std::remove_cvref_t<Input>;
      if constexpr (::ten::is_unary_expr<input_type>::value ||
                    ::ten::is_binary_expr<input_type>::value) {
         if (!_node->_input->evaluated()) {
            _node->_input->eval();
         }
      }

      // Evaluate
      if constexpr (::ten::is_scalar<input_type>::value ||
                    ::ten::is_tensor_v<input_type>) {
         _node->_func.value()(_node->_input, _node->_value);
      }

      if constexpr (::ten::is_unary_expr_v<input_type> ||
                    ::ten::is_binary_expr_v<input_type>) {
         // auto input = _node->_input->value();
         // auto output = _node->value();
         //_node->_func.value()(input, output);
         _node->_func.value()(_node->_input->value_node(), _node->value_node());
      }

      // This expression has been evaluated
      _node->_evaluated = true;

      return _node->value();
   }

   template <class Gradient>
   void backward_function(const std::optional<Gradient> &previous_grad) {
      using input_type = std::remove_cvref_t<Input>;
      if constexpr (::ten::is_scalar_v<input_type>) {
         auto input = *_node->_input.get();
         if (input.requires_grad()) {
            input.allocate_gradient();
            auto grad = input.grad();
            // Compute the gradient
            _node->_func.value().gradient(input, grad);
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
         }
      }
      if constexpr (::ten::is_tensor_v<input_type>) {
         auto input_tensor = *_node->_input.get();
         if (input_tensor.requires_grad()) {
            input_tensor.allocate_gradient();
            auto grad = input_tensor.grad();
            // Compute the gradient
            _node->_func.value().gradient(input_tensor, grad);
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
         }
      }
      if constexpr (::ten::is_unary_expr_v<input_type> ||
                    ::ten::is_binary_expr_v<input_type>) {
         if (has_retain_grad()) {
            _node->_input->value().allocate_gradient();
            auto grad = _node->_input->value().grad();
            // Compute the gradient
            _node->_func.value().gradient(_node->_input->value(), grad);
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_input->backward_function(opt_grad);
         } else {
            auto grad = ::ten::like(_node->_input->value());
            _node->_func.value().gradient(_node->_input->value(), grad);
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_input->backward_function(opt_grad);
         }
      }
   }

   void backward(bool create_graph = false) noexcept {
      if (!_node->_value) {
         std::cerr << "Called backward(create_graph = " << std::boolalpha
                   << create_graph << ") without calling eval first\n";
         return;
      }
      if (create_graph) {
         using input_type = std::remove_cvref_t<Input>;
         if constexpr (::ten::is_scalar_v<input_type>) {
            auto input = _node->input();
            if (!input.requires_grad()) {
               input.allocate_gradient();
            }
            auto grad = input.grad();
            _node->_func.value().gradient(input, grad);
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
         }
         if constexpr (::ten::is_tensor_v<input_type>) {
            auto input_tensor = _node->input();
            if (!input_tensor.requires_grad()) {
               input_tensor.allocate_gradient();
            }
            auto grad = input_tensor.grad();
            // Compute the gradient
            _node->_func.value().gradient(input_tensor, grad);
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
         }
         if constexpr (::ten::is_unary_expr_v<input_type> ||
                       ::ten::is_binary_expr_v<input_type>) {
            if (!_node->_input->value().requires_grad()) {
               _node->_input->value().allocate_gradient();
            }
            auto grad = _node->_input->value().grad();
            // Compute the gradient
            _node->_func.value().gradient(_node->_input->value(), grad);
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
            _node->_input->backward(create_graph);
         }
      } else {
         using Gradient = Output;
         std::optional<Gradient> grad(std::nullopt);
         backward_function(grad);
      }
   }
};

template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
struct binary_node {
   /// Left input type
   using left_type = typename details::output_type<Left>::type;

   /// Right input type
   using right_type = typename details::output_type<Right>::type;

   /// Output type
   using output_type = Output;

   /// Function type
   using func_type = Func<left_type, right_type, Output, Args...>;

   /// Flag for evaluated expression
   bool _evaluated = false;
   // Retain the gradient
   bool _retain_grad = false;
   /// Function
   std::optional<func_type> _func = std::nullopt;
   /// Left input
   std::shared_ptr<Left> _left = nullptr;
   /// Right input
   std::shared_ptr<Right> _right = nullptr;
   /// Output value
   std::shared_ptr<Output> _value = nullptr;

   /// Returns the left input
   [[nodiscard]] Left &left() const { return *_left.get(); }

   /// Returns a std::shared_ptr to the left input
   [[nodiscard]] std::shared_ptr<Left> left_node() const { return _left; }

   // Returns the right input
   [[nodiscard]] Right &right() const { return *_right.get(); }

   /// Returns a std::shared_ptr to the right input
   [[nodiscard]] std::shared_ptr<Right> right_node() const { return _right; }

   /// return the output
   [[nodiscard]] Output &value() const { return *_value.get(); }

   /// return a std::shared_ptr to the output
   [[nodiscard]] std::shared_ptr<Output> value_node() const { return _value; }

   /// Construct a binary noe if the function doesn't take additional
   /// parameters
   binary_node(Left &l, Right &r) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _left(std::make_shared<Left>(l)), _right(std::make_shared<Right>(r)),
         _func(func_type()) {}

   /// Construct a binary node if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   binary_node(Left &l, Right &r, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _func(func_type(std::forward<func_args>(fargs)...)),
         _left(std::make_shared<Left>(l)), _right(std::make_shared<Right>(r)) {}

   ~binary_node() {}
};

// \class binary_expr
// Binary expresion
// Left and Right can be scalar, tensor, row, column, unary_expr or binary_expr
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
class binary_expr : ten::expr<binary_expr<Left, Right, Output, Func, Args...>> {
 public:
   using left_ty = Left;
   using right_ty = Right;
   using output_ty = Output;

   /// Left input type
   using left_type = typename details::output_type<Left>::type;

   /// Right input type
   using right_type = typename details::output_type<Right>::type;

   /// Output type
   using output_type = Output;

   using func_type = Func<left_type, right_type, Output, Args...>;
   // using shape_type = typename Output::shape_type;

   using node_type = binary_node<Left, Right, Output, Func, Args...>;

   using value_type = output_type::value_type;

   using expr_type = binary_expr<Left, Right, Output, Func, Args...>;

 private:
   std::shared_ptr<node_type> _node = nullptr;

 public:
   binary_expr() {}

   ~binary_expr() {}

   /// Construct a binary expr if the function doesn't take additional
   /// parameters
   binary_expr(Left &l, Right &r) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
   {
      _node = std::make_shared<node_type>(l, r);
   }

   /// Construct a binary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   binary_expr(Left &l, Right &r, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
   {
      _node =
          std::make_shared<node_type>(l, r, std::forward<func_args>(fargs)...);
   }

   /// Requires gradient
   [[nodiscard]] inline bool requires_left_grad() {
      return _node->_left.requires_grad();
   }
   [[nodiscard]] inline bool requires_right_grad() {
      return _node->_right.requires_grad();
   }

   // Retain gradient
   void retain_grad() { _node->_retain_grad = true; }

   /// Returns whether the node has retain_grad
   [[nodiscard]] inline bool has_retain_grad() const {
      return _node->_retain_grad;
   }

   /// Returns the left input
   [[nodiscard]] Left &left() const { return *(_node->_left.get()); }

   /// Returns a std::shared_ptr to the left input
   [[nodiscard]] std::shared_ptr<Left> &left_node() const {
      return _node->_left;
   }

   // Returns the right input
   [[nodiscard]] Right &right() const { return *(_node->_right.get()); }

   /// Returns a std::shared_ptr to the right input
   [[nodiscard]] std::shared_ptr<Right> right_node() const {
      return _node->_right;
   }

   // Left gradient
   [[nodiscard]] auto left_grad() const {
      if constexpr (::ten::is_tensor_v<Left> || ::ten::is_scalar_v<Left>) {
         return _node->_left->grad();
      } else {
         return _node->_left->value().grad();
      }
   }

   // Right gradient
   [[nodiscard]] auto right_grad() const {
      if constexpr (::ten::is_tensor_v<Right> || ::ten::is_scalar_v<Right>) {
         return _node->_right->grad();
      } else {
         return _node->_right->value().grad();
      }
   }

   /// Returns whether the expression is evaluated
   [[nodiscard]] inline bool evaluated() const { return _node->_evaluated; }

   /// Returns the the evaluated expression of type ten::scalar or ten::tensor
   [[nodiscard]] output_type &value() { return *(_node->_value.get()); }

   /// Returns the the std::shared_ptr to the evaluated expression of type
   /// ten::scalar or ten::tensor
   [[nodiscard]] output_type &value_node() { return _node->_value; }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (evaluated())
         return _node->value();

      // Evaluate the left expr
      using left_type = std::remove_cvref_t<Left>;
      if constexpr (::ten::is_unary_expr_v<left_type> ||
                    ::ten::is_binary_expr_v<left_type>) {
         if (!_node->_left->evaluated()) {
            _node->_left->eval();
         }
      }
      // Evaluate the right expr
      using right_type = std::remove_cvref_t<Right>;
      if constexpr (::ten::is_unary_expr_v<right_type> ||
                    ::ten::is_binary_expr_v<right_type>) {
         if (!_node->_right->evaluated()) {
            _node->_right->eval();
         }
      }

      // Allocate the output if it has not been allocated yet
      /*
      if (!_node->_value) {
         if constexpr (::ten::is_scalar_v<Output>) {
            _node->_value = std::make_shared<Output>();
         } else if constexpr (Output::is_static()) {
            _node->_value = std::make_shared<Output>();
         } else if constexpr (::ten::is_unary_expr<Left>::value &&
                              ::ten::is_unary_expr<Right>::value) {
            // Left and Right are unary_expr
            // They can't be both scalars
            if constexpr (::ten::is_scalar<left_type>::value &&
                          !::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->right())));
            }
            if constexpr (!::ten::is_scalar<left_type>::value &&
                          ::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left())));
            }
            if constexpr (!::ten::is_scalar<left_type>::value &&
                          !::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else if constexpr (::ten::is_binary_expr<Left>::value &&
                              ::ten::is_binary_expr<Right>::value) {
            // Left and Right are binary_expr
            if constexpr (::ten::is_scalar<left_type>::value &&
                          !::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->right())));
            }
            if constexpr (!::ten::is_scalar<left_type>::value &&
                          ::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left())));
            }
            if constexpr (!::ten::is_scalar<left_type>::value &&
                          !::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else if constexpr (::ten::is_unary_expr<Left>::value &&
                              !::ten::is_unary_expr<Right>::value &&
                              !::ten::is_binary_expr<Right>::value) {
            // Left is unary expr and right is tensor or scalar
            if constexpr (::ten::is_scalar<left_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->right())));
            } else {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else if constexpr (::ten::is_binary_expr_v<Left> &&
                              !ten::is_unary_expr_v<Right> &&
                              !::ten::is_binary_expr_v<Right>) {
            // Left is binary expr and right is tensor or scalar
            if constexpr (::ten::is_scalar<left_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->right())));
            } else {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else if constexpr (!::ten::is_unary_expr<Left>::value &&
                              !::ten::is_binary_expr<Left>::value &&
                              ::ten::is_unary_expr<Right>::value) {
            // Left is tensor or scalar and Right is unary_expr
            if constexpr (::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left())));
            } else {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else if constexpr (!::ten::is_unary_expr<Left>::value &&
                              !::ten::is_binary_expr<Left>::value &&
                              ten::is_binary_expr<Right>::value) {
            // Left is tensor or scalar and right is binary_expr
            if constexpr (::ten::is_scalar<right_type>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left())));
            } else {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
         } else {
            // Left and right are both tensor or scalar
            if constexpr (!::ten::is_scalar<Left>::value &&
                          !::ten::is_scalar<Right>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left()),
                   ::ten::details::input_shape(_node->right())));
            }
            if constexpr (::ten::is_scalar<Left>::value &&
                          !::ten::is_scalar<Right>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->right())));
            }
            if constexpr (!::ten::is_scalar<Left>::value &&
                          ::ten::is_scalar<Right>::value) {
               _node->_value = std::make_shared<Output>(func_type::output_shape(
                   ::ten::details::input_shape(_node->left())));
            }
         }
      }
      */

      // Call the function
      _node->_func.value()(::ten::details::input_value(_node->left()),
                           ::ten::details::input_value(_node->right()),
                           _node->value());

      // This expression has been evaluated
      _node->_evaluated = true;

      return _node->value();
   }

   template <class Gradient>
   void backward_function(std::optional<Gradient> &previous_grad) {
      using left_type = std::remove_cvref_t<Left>;
      using right_type = std::remove_cvref_t<Right>;
      // Left is tensor
      if constexpr (::ten::is_tensor_v<left_type>) {
         auto left_tensor = _node->left();
         if (left_tensor.requires_grad()) {
            left_tensor.allocate_gradient();
            auto grad = left_tensor.grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<right_type>) {
               _node->_func.value().gradient_left(left_tensor, _node->right(),
                                                  grad);
            } else {
               _node->_func.value().gradient_left(left_tensor,
                                                  _node->_right->value(), grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
         }
      }
      // Right is tensor
      if constexpr (::ten::is_tensor_v<right_type>) {
         auto right_tensor = _node->right();
         if (right_tensor.requires_grad()) {
            // Allocate the gradient for the right tensor
            right_tensor.allocate_gradient();
            auto grad = right_tensor.grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<left_type>) {
               _node->_func.value().gradient_right(_node->left(), right_tensor,
                                                   grad);
            } else {
               _node->_func.value().gradient_right(_node->_left->value(),
                                                   right_tensor, grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
         }
      }
      // Left is expr
      if constexpr (::ten::is_unary_expr_v<left_type> ||
                    ::ten::is_binary_expr_v<left_type>) {
         if (_node->_left->has_retain_grad()) {
            _node->_left->value().allocate_gradient();
            auto grad = _node->_left->value().grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<right_type>) {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->right(), grad);
            } else {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->_right->value(), grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_left->backward_function(opt_grad);
         } else {
            auto grad = ::ten::like(_node->_left->value());
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<right_type>) {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->right(), grad);
            } else {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->_right->value(), grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_left->backward_function(opt_grad);
         }
      }
      // Right is expr
      if constexpr (::ten::is_unary_expr_v<right_type> ||
                    ::ten::is_binary_expr_v<right_type>) {
         if (_node->_right->has_retain_grad()) {
            _node->_right->value().allocate_gradient();
            auto grad = _node->_right->value().grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<left_type>) {
               _node->_func.value().gradient_right(
                   _node->left(), _node->_right->value(), grad);
            } else {
               _node->_func.value().gradient_right(
                   _node->_left->value(), _node->_right->value(), grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_right->backward_function(opt_grad);
         } else {
            auto grad = ::ten::like(_node->_right->value());
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<left_type>) {
               _node->_func.value().gradient_right(
                   _node->left(), _node->_right->value(), grad);
            } else {
               _node->_func.value().gradient_right(
                   _node->_left->value(), _node->_right->value(), grad);
            }
            // Use the chain rule
            if (previous_grad.has_value()) {
               backward_chain(grad, previous_grad.value());
            }
            using grad_type = std::remove_cvref_t<decltype(grad)>;
            std::optional<grad_type> opt_grad(grad);
            _node->_right->backward_function(opt_grad);
         }
      }
   }

   void backward(bool create_graph = false) noexcept {
      if (!_node->_value) {
         std::cerr << "Called backward(create_graph = " << std::boolalpha
                   << create_graph << ") without calling eval first\n";
         return;
      }
      using left_type = std::remove_cvref_t<Left>;
      using right_type = std::remove_cvref_t<Right>;
      if (create_graph) {
         // Left is tensor
         if constexpr (::ten::is_tensor_v<left_type>) {
            auto left_tensor = _node->left();
            if (!left_tensor.requires_grad()) {
               left_tensor.allocate_gradient();
            }
            auto grad = left_tensor.grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<right_type>) {
               _node->_func.value().gradient_left(left_tensor, _node->right(),
                                                  grad);
            } else {
               _node->_func.value().gradient_left(left_tensor,
                                                  _node->_right->value(), grad);
            }
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
         }
         // Right is tensor
         if constexpr (::ten::is_tensor_v<right_type>) {
            auto right_tensor = _node->right();
            if (!right_tensor.requires_grad()) {
               right_tensor.allocate_gradient();
            }
            auto grad = right_tensor.grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<left_type>) {
               _node->_func.value().gradient_right(_node->left(), right_tensor,
                                                   grad);
            } else {
               _node->_func.value().gradient_right(_node->_left->value(),
                                                   right_tensor, grad);
            }
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
         }
         // Left is expr
         if constexpr (::ten::is_unary_expr_v<left_type> ||
                       ::ten::is_binary_expr_v<left_type>) {
            if (!_node->_left->value().requires_grad()) {
               _node->_left->value().allocate_gradient();
            }
            auto grad = _node->_left->value().grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<right_type>) {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->right(), grad);
            } else {
               _node->_func.value().gradient_left(_node->_left->value(),
                                                  _node->_right->value(), grad);
            }
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
            _node->_left->backward(create_graph);
         }
         // Right is expr
         if constexpr (::ten::is_unary_expr_v<right_type> ||
                       ::ten::is_binary_expr_v<right_type>) {
            if (!_node->_right->value().requires_grad()) {
               _node->_right->value().allocate_gradient();
            }
            auto grad = _node->_right->value().grad();
            // Compute the gradient
            if constexpr (::ten::is_tensor_v<left_type>) {
               _node->_func.value().gradient_right(
                   _node->left(), _node->_right->value(), grad);
            } else {
               _node->_func.value().gradient_right(
                   _node->_left->value(), _node->_right->value(), grad);
            }
            // Use the chain rule
            if (_node->_value->requires_grad()) {
               auto previous_grad = _node->_value->grad();
               backward_chain(grad, previous_grad);
            }
            _node->_right->backward(create_graph);
         }
      } else {
         using Gradient = Output;
         // auto output = _node->value();
         std::optional<Gradient> grad(std::nullopt);
         backward_function(grad);
      }
   }
};

} // namespace ten

#endif
