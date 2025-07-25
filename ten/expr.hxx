#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <ten/functional.hxx>
#include <ten/types.hxx>

namespace ten::details {

// Input type
template <class> struct input_type;

template <class T> struct input_type<scalar<T>> {
   using type = scalar<T>;
};

template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
struct input_type<
    ranked_tensor<T, Shape, order, Storage, Allocator>> {
   using type = ranked_tensor<T, Shape, order, Storage, Allocator>;
};

template <class Input, class Output, template <typename...> class F,
          class... Args>
struct input_type<::ten::unary_expr<Input, Output, F, Args...>> {
   using type = typename ::ten::unary_expr<Input, Output, F,
                                           Args...>::input_type;
};

template <class Left, class Right, class Output,
          template <typename...> class F, class... Args>
struct input_type<
    ::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::input_type;
};

// Output type
template <class> struct output_type;

template <class T> struct output_type<scalar<T>> {
   using type = scalar<T>;
};

template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
struct output_type<
    ranked_tensor<T, Shape, order, Storage, Allocator>> {
   using type = ranked_tensor<T, Shape, order, Storage, Allocator>;
};

template <class Input, class Output, template <typename...> class F,
          class... Args>
struct output_type<::ten::unary_expr<Input, Output, F, Args...>> {
   using type = typename ::ten::unary_expr<Input, Output, F,
                                           Args...>::output_type;
};

template <class Left, class Right, class Output,
          template <typename...> class F, class... Args>
struct output_type<
    ::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::output_type;
};

// Input shape
template<Tensor T>
inline auto input_shape(const T& t) -> decltype(auto) {
   return t.shape();
}

template<UnaryExpr ExprType>
inline auto input_shape(const ExprType& expr) -> decltype(auto) {
   return expr.value().shape();
}

template<BinaryExpr ExprType>
inline auto input_shape(const ExprType& expr) -> decltype(auto) {
   return expr.value().shape();
}

// Input value
template<Tensor T>
inline auto input_value(T& t) {return t;}

template<UnaryExpr ExprType>
inline auto input_value(ExprType& expr) {return expr.value();}

template<BinaryExpr ExprType>
inline auto input_value(ExprType& expr) {return expr.value();}

}

namespace ten {

// \class unary_expr
// Apply a function to a tensor, column or row
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class unary_expr : ten::expr<unary_expr<Input, Output, Func, Args...>>{
 public:
   using input_type = typename ::ten::details::input_type<Input>::type;
   using output_type = typename ::ten::details::output_type<Output>::type;

   using func_type = Func<input_type, output_type>;

 private:
   /// Flag for evaluated expression
   bool _evaluated = false;
   /// Optional function type
   std::optional<func_type> _func = std::nullopt;
   //// Output value
   Input _input;
   Output _value;

 public:
   unary_expr() {}

   /// Construct a unary expr if the function doesn't take additional parameters
   unary_expr(const Input &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type()) {}

   /// Construct a unary expr if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_expr(const Input &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type(std::forward<func_args>(fargs)...)) {}

   /// Returns a shared ptr to the output
   /*[[nodiscard]] inline Output output() { return _value; }*/

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _evaluated; }

   /// Returns the evaluated expression of type ten::Scalar or ten::Tensor
   [[nodiscard]] auto value() const -> output_type {
      return _value;
   }

   /// Evaluate the expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (_evaluated)
         return _value;

      // Evaluate input
      if constexpr (::ten::is_unary_expr<Input>::value ||
                    ::ten::is_binary_expr<Input>::value) {
         if (_input.evaluated()) {
            _input.eval();
         }
      }

      // Allocate output
      if constexpr (::ten::is_scalar<Output>::value) {
         _value = Output();
      } else if constexpr (!::ten::is_scalar<Output>::value &&
                           Output::is_static()) {
         _value = Output();
      } else {
         if constexpr (ten::functional::has_shape<func_type>::value) {
            _value = Output(_func.value().output_shape(::ten::details::input_shape(_input)));
         }
         if constexpr (!ten::functional::has_shape<func_type>::value) {
            _value = Output(_func.value().output_shape(::ten::details::input_shape(_input)));
         }
      }

      // Evaluate
      _func.value()(::ten::details::input_value(_input), _value);
      _evaluated = true;
      return _value;
   }
};

// \class binary_expr
// Binary expresion
// Left and Right can be scalar, tensor, row, column, unary_expr or binary_expr
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
class binary_expr {
 public:
   /// Left input type
   using left_type = typename details::output_type<Left>::type;

   /// Right input type
   using right_type = typename details::output_type<Right>::type;

   /// Output type
   using output_type = Output;

   using func_type =
       Func<left_type, right_type, Output, Args...>;
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

   /// Construct a binary expr if the function doesn't take additional parameters
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

   /// Returns the shared ptr to the output node
   //[[nodiscard]] inline std::shared_ptr<Output> node() { return _value; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _evaluated; }

   /// Returns the the evaluated expression of type ten::Scalar or ten::Tensor
   [[nodiscard]] auto value() -> output_type {
      return _value;
   }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() noexcept -> output_type {
      if (_evaluated)
         return _value;

      // Evaluate the left expr
      if constexpr (::ten::is_unary_expr<Left>::value ||
                    ::ten::is_binary_expr<Left>::value) {
         if (!_left.evaluated()) {
            _left.eval();
         }
      }

      // Evaluate the right expr
      if constexpr (::ten::is_unary_expr<Right>::value ||
                    ::ten::is_binary_expr<Right>::value) {
         if (!_right.evaluated()) {
            _right.eval();
         }
      }

      if constexpr (Output::is_static()) {
         _value = Output();
         // FIXME if right is unary expr
      } else if constexpr (::ten::is_unary_expr<Left>::value) {
         if constexpr (::ten::is_scalar<Left>::value) {
            _value = Output(func_type::output_shape(::ten::details::input_shape(_right)));
         }
      } else {
         // FIXME May requires using ten::functional::has_shape when
         // a binary function has its own shape
         if constexpr (!::ten::is_scalar<Left>::value && !::ten::is_scalar<Right>::value) {
            _value = Output(func_type::output_shape(
                ::ten::details::input_shape(_left),
                ::ten::details::input_shape(_right)));
         }
         if constexpr (::ten::is_scalar<Left>::value && !::ten::is_scalar<Right>::value) {
            _value = Output(func_type::output_shape(::ten::details::input_shape(_right)));
         }
         if constexpr (!::ten::is_scalar<Left>::value && ::ten::is_scalar<Right>::value) {
            _value = Output(func_type::output_shape(::ten::details::input_shape(_left)));
         }
      }

      // Call the function
      _func.value()(::ten::details::input_value(_left), ::ten::details::input_value(_right), _value);

      // This expression has been evaluated
      _evaluated = true;

      return _value;
   }
};

} // namespace ten

#endif
