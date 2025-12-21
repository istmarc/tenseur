#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <ios>
#include <memory>
#include <optional>
#include <type_traits>

#include <ten/functional.hxx>
#include <ten/types.hxx>

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
   using type =
       typename ::ten::binary_expr<Left, Right, Output, F, Args...>::input_type;
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
template <Tensor T>
static inline auto input_shape(T &t) -> decltype(auto) {
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
template<class Input, class Output, template<typename...> class Func, typename ...Args>
struct unary_node {
   using input_type = typename ::ten::details::input_type<Input>::type;
   using output_type = typename ::ten::details::output_type<Output>::type;
   using func_type = Func<input_type, output_type>;

   bool _evaluated = false;
   std::optional<func_type> _func = std::nullopt;
   Input _input;
   Output _value;

   unary_node(Input &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type()) {}

   /// Construct a unary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_node(Input &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type(std::forward<func_args>(fargs)...)) {}
};

// \class unary_expr
// Apply a function to a tensor, column or row
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class unary_expr : ten::expr<unary_expr<Input, Output, Func, Args...>> {
 public:
   using input_ty = Input;
   using output_ty = Output;

   using input_type = typename ::ten::details::input_type<Input>::type;
   using output_type = typename ::ten::details::output_type<Output>::type;

   using func_type = Func<input_type, output_type>;

   using node_type = unary_node<Input, Output, Func, Args...>;

 private:
   std::shared_ptr<node_type> _node = nullptr;

 public:
   unary_expr() {}

   /// Construct a unary expr if the function doesn't take additional parameters
   unary_expr(Input &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value) {
      _node = std::make_shared<node_type>(inp);
   }

   /// Construct a unary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_expr(Input &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value) {
         _node = std::make_shared<node_type>(inp, std::forward<func_args>(fargs)...);
   }

   /// Returns a shared ptr to the output
   /*[[nodiscard]] inline Output output() { return _value; }*/

   /// Requires gradient
   [[nodiscard]] inline bool requires_grad() {return _node->_input.requires_grad();}

   /// Return the input
   [[nodiscard]] inline Input &input() { return _node->_input; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] inline bool evaluated() { return _node->_evaluated; }

   /// Returns the evaluated expression of type ten::Scalar or ten::Tensor
   [[nodiscard]] inline output_type& value() { return _node->_value; }


   [[nodiscard]] auto grad() noexcept -> output_type {
      using input_type = std::remove_cvref_t<Input>;
      // TODO if the input is scalar
      //if constexpr (::ten::is_scalar_v<input_type>) {
      //   return _input.grad();
      //}
      if constexpr (::ten::is_tensor_v<input_type>) {
         return _node->_input.grad();
      }
      if constexpr (::ten::is_unary_expr_v<input_type> || ::ten::is_binary_expr_v<input_type>) {
         return _node->_input.value().grad();
      }
   }

   /// Evaluate the expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (evaluated())
         return _node->_value;

      // Evaluate input
      using input_type = std::remove_cvref_t<Input>;
      if constexpr (::ten::is_unary_expr<input_type>::value ||
                    ::ten::is_binary_expr<input_type>::value) {
         if (!_node->_input.evaluated()) {
            _node->_input.eval();
         }
      }

      // Allocate output
      if constexpr (::ten::is_scalar<output_type>::value) {
         _node->_value = Output();
      } else if constexpr (!::ten::is_scalar<output_type>::value &&
                           Output::is_static()) {
         _node->_value = Output();
      } else {
         if constexpr (ten::functional::has_shape<func_type>::value) {
            _node->_value = Output(_node->_func.value().output_shape(
                ::ten::details::input_shape(_node->_input)));
         }
         if constexpr (!ten::functional::has_shape<func_type>::value) {
            _node->_value = Output(_node->_func.value().output_shape(
                ::ten::details::input_shape(_node->_input)));
         }
      }

      // Evaluate
      //_func.value()(::ten::details::input_value(_input), _value);
      if constexpr (::ten::is_scalar<input_type>::value || ::ten::is_tensor_v<input_type>) {
         _node->_func.value()(_node->_input, _node->_value);
      }
      if constexpr (::ten::is_unary_expr_v<input_type> || ::ten::is_binary_expr_v<input_type>) {
         _node->_func.value()(_node->_input.value(), _node->_value);
      }

      // This expression has been evaluated
      _node->_evaluated = true;

      return _node->_value;
   }

   [[maybe_unused]] auto backward() noexcept -> output_type {
      using input_type = std::remove_cvref_t<Input>;
      // if the input is a tensor
      if constexpr (::ten::is_tensor_v<input_type>) {
         compute_gradient<func_type>(_node->_input);
         // return output_type(_node->_input.grad_node(), _node->_input.shape(), _node->_input.format(), false);
         return _node->_input.grad();
      }
      // If the input is a unary node
      // Compute gradient of fog = f' x g'of
      if constexpr (::ten::is_unary_expr_v<input_type>) {
         // comput g' in _input.grad
         _node->_input.backward();
         // Compute f' in and g'of
         compute_gradient_chain_rule<func_type>(_node->_input.value(), _node->_input.grad());
         return _node->_input.value().grad();
      }
      // TODO If the input is a binary node
   }

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

 private:
   /// Flag for evaluated expression
   bool _evaluated = false;
   /// Function
   std::optional<func_type> _func = std::nullopt;
   /// Left input
   Left _left;
   /// Right input
   Right _right;
   /// Output value
   Output _value;

 public:
   binary_expr() {}

   /// Construct a binary expr if the function doesn't take additional
   /// parameters
   binary_expr(const Left &l, const Right &r) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _left(l), _right(r), _func(func_type()) {}

   /// Construct a binary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   binary_expr(const Left &l, const Right &r, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _func(func_type(std::forward<func_args>(fargs)...)), _left(l),
         _right(r) {}

   /// Returns the left input
   [[nodiscard]] const Left &left() const { return _left; }

   // Returns the right input
   [[nodiscard]] const Right &right() const { return _right; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _evaluated; }

   /// Returns the the evaluated expression of type ten::Scalar or ten::Tensor
   [[nodiscard]] auto value() const -> output_type { return _value; }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (_evaluated)
         return _value;

      // Evaluate the left expr
      if constexpr (::ten::is_unary_expr_v<Left> ||
                    ::ten::is_binary_expr_v<Left>) {
         if (!_left.evaluated()) {
            _left.eval();
         }
      }

      // Evaluate the right expr
      if constexpr (::ten::is_unary_expr_v<Right> ||
                    ::ten::is_binary_expr_v<Right>) {
         if (!_right.evaluated()) {
            _right.eval();
         }
      }

      if constexpr (Output::is_static()) {
         _value = Output();
      } else if constexpr (::ten::is_unary_expr<Left>::value &&
                           ::ten::is_unary_expr<Right>::value) {
         // Left and Right are unary_expr
         // They can't be both scalars
         if constexpr (::ten::is_scalar<left_type>::value &&
                       !::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_right)));
         }
         if constexpr (!::ten::is_scalar<left_type>::value &&
                       ::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left)));
         }
         if constexpr (!::ten::is_scalar<left_type>::value &&
                       !::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else if constexpr (::ten::is_binary_expr<Left>::value &&
                           ::ten::is_binary_expr<Right>::value) {
         // Left and Right are binary_expr
         if constexpr (::ten::is_scalar<left_type>::value &&
                       !::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_right)));
         }
         if constexpr (!::ten::is_scalar<left_type>::value &&
                       ::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left)));
         }
         if constexpr (!::ten::is_scalar<left_type>::value &&
                       !::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else if constexpr (::ten::is_unary_expr<Left>::value &&
                           !::ten::is_unary_expr<Right>::value &&
                           !::ten::is_binary_expr<Right>::value) {
         // Left is unary expr and right is tensor or scalar
         if constexpr (::ten::is_scalar<left_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_right)));
         } else {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else if constexpr (::ten::is_binary_expr_v<Left> &&
                           !ten::is_unary_expr_v<Right> &&
                           !::ten::is_binary_expr_v<Right>) {
         // Left is binary expr and right is tensor or scalar
         if constexpr (::ten::is_scalar<left_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_right)));
         } else {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else if constexpr (!::ten::is_unary_expr<Left>::value &&
                           !::ten::is_binary_expr<Left>::value &&
                           ::ten::is_unary_expr<Right>::value) {
         // Left is tensor or scalar and Right is unary_expr
         if constexpr (::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left)));
         } else {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else if constexpr (!::ten::is_unary_expr<Left>::value &&
                           !::ten::is_binary_expr<Left>::value &&
                           ten::is_binary_expr<Right>::value) {
         // Left is tensor or scalar and right is binary_expr
         if constexpr (::ten::is_scalar<right_type>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left)));
         } else {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
      } else {
         // Left and right are both tensor or scalar
         if constexpr (!::ten::is_scalar<Left>::value &&
                       !::ten::is_scalar<Right>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left),
                                        ::ten::details::input_shape(_right)));
         }
         if constexpr (::ten::is_scalar<Left>::value &&
                       !::ten::is_scalar<Right>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_right)));
         }
         if constexpr (!::ten::is_scalar<Left>::value &&
                       ::ten::is_scalar<Right>::value) {
            _value = Output(
                func_type::output_shape(::ten::details::input_shape(_left)));
         }
      }

      // Call the function
      _func.value()(::ten::details::input_value(_left),
                    ::ten::details::input_value(_right), _value);

      // This expression has been evaluated
      _evaluated = true;

      return _value;
   }
};

} // namespace ten

#endif
