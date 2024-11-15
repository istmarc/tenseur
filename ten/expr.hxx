#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <ten/functional.hxx>
#include <ten/types.hxx>

namespace ten {

// FIXME This may be useful for getting Functional::Apply to work?
// Keep it here for now. F is the function type or Inner Function type
// Expression must accept a function_wrapper<...> or
// function_wrapper<..>::outer_function_wrapper<...>
template <template <class...> class f, class... args> struct function_wrapper {
   template <template <template <class...> class /*F*/, class... /*Args*/>
             class g>
   struct outer_function_wrapper {};
};

namespace details {
// Node shape
template <class> struct node_wrapper;

template <class __t> struct node_wrapper<scalar_node<__t>> {
   static auto ptr(const std::shared_ptr<scalar_node<__t>> &node) {
      return node.get();
   }
};

template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
struct node_wrapper<
    tensor_node<__t, __shape, __order, __storage, __allocator>> {
   static auto
   shape(const std::shared_ptr<
         tensor_node<__t, __shape, __order, __storage, __allocator>> &node) {
      return node.get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<
       tensor_node<__t, __shape, __order, __storage, __allocator>> &node) {
      return node.get();
   }
};

template <class __input, class __output, template <typename...> class __f,
          typename... __args>
struct node_wrapper<unary_node<__input, __output, __f, __args...>> {
   static auto
   shape(const std::shared_ptr<unary_node<__input, __output, __f, __args...>>
             &node) {
      return node.get()->node().get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<unary_node<__input, __output, __f, __args...>>
           &node) {
      return node.get()->node().get();
   }
};

template <class __left, class __right, class __output,
          template <typename...> class __f, typename... __args>
struct node_wrapper<binary_node<__left, __right, __output, __f, __args...>> {
   static auto
   shape(const std::shared_ptr<
         binary_node<__left, __right, __output, __f, __args...>> &node) {
      return node.get()->node().get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<
       binary_node<__left, __right, __output, __f, __args...>> &node) {
      return node.get()->node().get();
   }
};
} // namespace details

namespace details {
// Node type
template <class> struct output_node_type;

template <class __t> struct output_node_type<scalar_node<__t>> {
   using type = scalar_node<__t>;
};

template <class __t, class __shape, storage_order __order, class __storage,
          class __allocator>
struct output_node_type<
    tensor_node<__t, __shape, __order, __storage, __allocator>> {
   using type = tensor_node<__t, __shape, __order, __storage, __allocator>;
};

template <class __input, class __output, template <typename...> class __f,
          class... __args>
struct output_node_type<::ten::unary_node<__input, __output, __f, __args...>> {
   using type = typename ::ten::unary_node<__input, __output, __f,
                                           __args...>::output_node_type;
};

template <class __left, class __right, class __output,
          template <typename...> class __f, class... __args>
struct output_node_type<
    ::ten::binary_node<__left, __right, __output, __f, __args...>> {
   using type = typename ::ten::binary_node<__left, __right, __output, __f,
                                            __args...>::output_node_type;
};
} // namespace details

// \class unary_node
// Apply a function to a ten::Tensor or a ten::Scalar
template <class __input, class __output, template <typename...> class __func,
          typename... __args>
class unary_node {
 public:
   using input_node_type = typename details::output_node_type<__input>::type;
   using output_node_type = __output;
   using func_type = __func<input_node_type, __output>;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<is_scalar_node<__output>::value,
                                             typename __output::scalar_type,
                                             typename __output::tensor_type>;

 private:
   /// Optional function type
   std::optional<func_type> _func = std::nullopt;
   //// Output value
   std::shared_ptr<__output> _value = nullptr;
   std::shared_ptr<__input> _input = nullptr;

 public:
   unary_node() {}

   /// Construct a unary node if the function doesn't take additional parameters
   unary_node(const std::shared_ptr<__input> &inp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type()) {}

   /// Construct a unary node if the function take additional parameters.
   /// The parameters of the functions fargs of type func_args are forwarded
   /// to the constructor of the function when necessary.
   template <typename... func_args>
   unary_node(const std::shared_ptr<__input> &inp, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _input(inp), _func(func_type(std::forward<func_args>(fargs)...)) {}

   /// Returns a shared ptr to the output node
   [[nodiscard]] inline std::shared_ptr<__output> node() { return _value; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _value.get(); }

   /// Returns the evaluated expression of type ten::Scalar or ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_value);
   }

   /// Evaluated the expression
   [[maybe_unused]] auto eval() noexcept -> evaluated_type {
      if (_value)
         return evaluated_type(_value);

      // Evaluate input
      if constexpr (::ten::is_unary_node<__input>::value ||
                    ::ten::is_binary_node<__input>::value) {
         if (!_input.get()->evaluated()) {
            _input.get()->eval();
         }
      }

      // Allocate output
      if constexpr (::ten::is_scalar_node<__output>::value) {
         _value.reset(new __output());
      } else if constexpr (!::ten::is_scalar_node<__output>::value &&
                           __output::is_static()) {
         _value.reset(new __output());
      } else {
         if constexpr (ten::functional::has_shape<func_type>::value) {
            _value.reset(new __output(_func.value().output_shape(
                ::ten::details::node_wrapper<__input>::shape(_input))));
         }
         if constexpr (!ten::functional::has_shape<func_type>::value) {
            _value.reset(new __output(func_type::output_shape(
                ::ten::details::node_wrapper<__input>::shape(_input))));
         }
      }

      // Evaluate
      _func.value()(*::ten::details::node_wrapper<__input>::ptr(_input),
                    *_value.get());

      return evaluated_type(_value);
   }
};

/// \class unary_expr
/// Unary expression.
template <typename __expr, template <class...> class __func, typename... __args>
class unary_expr : ::ten::expr<unary_expr<__expr, __func, __args...>> {
 public:
   /// \typedef input_type
   /// Type of the input type of the function
   using input_node_type =
       typename ::ten::details::output_node_type<__expr>::type;

   /// \typedef output_type
   /// Type of the output type of the function
   using output_node_type = typename __func<input_node_type>::output_type;

   /// \typedef node_type
   /// Type of the node of the unary expresion
   using node_type = unary_node<__expr, output_node_type, __func, __args...>;

   /// \typedef expr_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<is_scalar_node<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

   using func_type = __func<input_node_type>;

 private:
   /// Shared ptr to the node
   std::shared_ptr<node_type> _node;

 public:
   /// Construct a ten::unary_node from an expression
   explicit unary_expr(const std::shared_ptr<__expr> &exp) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _node(std::make_shared<node_type>(exp)) {}

   /// Construct a ten::unary_node from an expression with a parametric function
   template <typename... func_args>
   explicit unary_expr(const std::shared_ptr<__expr> &exp,
                       func_args &&...fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _node(std::make_shared<node_type>(exp,
                                           std::forward<func_args>(fargs)...)) {
   }

   // Returns the shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _node.get()->evaluated(); }

   /// Returns the scalar or tensor value of the evaluated expression
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_node.get()->node());
   }

   /// Evaluate a unary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression.
   [[maybe_unused]] auto eval() -> evaluated_type {
      return _node.get()->eval();
   }
};

// \class binary_node
// Node of a binary expresion
// Left and Right can be scalar_node, tensor_node or binary_node
template <class __left, class __right, class __output,
          template <typename...> class __func, typename... __args>
class binary_node {
 public:
   /// Left input type
   using left_node_type = typename details::output_node_type<__left>::type;

   /// Right input type
   using right_node_type = typename details::output_node_type<__right>::type;

   /// Output type
   using output_node_type = __output;

   using func_type =
       __func<left_node_type, right_node_type, __output, __args...>;
   // using shape_type = typename Output::shape_type;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<is_scalar_node<__output>::value,
                                             typename __output::scalar_type,
                                             typename __output::tensor_type>;

 private:
   std::optional<func_type> _func = std::nullopt;
   // std::optional<typename output::shape_type> _shape = std::nullopt;
   std::shared_ptr<__output> _value = nullptr;
   std::shared_ptr<__left> _left;
   std::shared_ptr<__right> _right;

 public:
   binary_node() {}

   binary_node(const std::shared_ptr<__left> &l,
               const std::shared_ptr<__right> &r) noexcept
      requires(!::ten::functional::has_params<func_type>::value)
       : _left(l), _right(r), _func(func_type()) {}

   template <typename... func_args>
   binary_node(const std::shared_ptr<__left> &l,
               const std::shared_ptr<__right> &r, func_args... fargs) noexcept
      requires(::ten::functional::has_params<func_type>::value)
       : _func(func_type(std::forward<func_args>(fargs)...)), _left(l),
         _right(r) {}

   /// Returns the shared ptr to the output node
   [[nodiscard]] inline std::shared_ptr<__output> node() { return _value; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _value.get(); }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_value);
   }

   [[maybe_unused]] auto eval() noexcept -> evaluated_type {
      if (_value)
         return evaluated_type(_value);

      // Evaluate the left expr
      if constexpr (::ten::is_unary_node<__left>::value ||
                    ::ten::is_binary_node<__left>::value) {
         if (!_left.get()->evaluated()) {
            _left.get()->eval();
         }
      }

      // Evaluate the right expr
      if constexpr (::ten::is_unary_node<__right>::value ||
                    ::ten::is_binary_node<__right>::value) {
         if (!_right.get()->evaluated()) {
            _right.get()->eval();
         }
      }

      if constexpr (__output::is_static()) {
         _value.reset(new __output());
      } else {
         // FIXME May requires using ten::functional::has_shape when
         // a binary function has its own shape
         if constexpr (!::ten::is_scalar_node<__left>::value &&
                       !::ten::is_scalar_node<__right>::value) {
            _value.reset(new __output(func_type::output_shape(
                ::ten::details::node_wrapper<__left>::shape(_left),
                ::ten::details::node_wrapper<__right>::shape(_right))));
         } else {
            if constexpr (::ten::is_scalar_node<__left>::value &&
                          !::ten::is_scalar_node<__right>::value) {
               _value.reset(new __output(func_type::output_shape(
                   ::ten::details::node_wrapper<__right>::shape(_right))));
            }
            if constexpr (!::ten::is_scalar_node<__left>::value &&
                          ::ten::is_scalar_node<__right>::value) {
               _value.reset(new __output(func_type::output_shape(
                   ::ten::details::node_wrapper<__left>::shape(_left))));
            }
         }
      }

      // Call the function
      _func.value()(*details::node_wrapper<__left>::ptr(_left),
                    *details::node_wrapper<__right>::ptr(_right),
                    *_value.get());

      return evaluated_type(_value);
   }
};

/// \class binary_expr
/// Binary expression
// left and right can be scalar_node, tensor_node or binary_expr
// left is std::shared_ptr<__left> and right is std::shared_ptr<__right>
template <typename __left, typename __right,
          template <typename...> class __func, typename... __args>
class binary_expr
    : ::ten::expr<binary_expr<__left, __right, __func, __args...>> {
 public:
   /// Left input type
   using left_node_type = typename details::output_node_type<__left>::type;

   /// Right input type
   using right_node_type = typename details::output_node_type<__right>::type;

   /// output_node_type is scalar_node or tensor_node
   using output_node_type =
       typename __func<left_node_type, right_node_type>::output_type;

   // Node type
   using node_type =
       binary_node<__left, __right, output_node_type, __func, __args...>;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<is_scalar_node<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

 private:
   std::shared_ptr<node_type> _node;

 public:
   /// Construct a binary_expr from an expression
   explicit binary_expr(const std::shared_ptr<__left> &l,
                        const std::shared_ptr<__right> &r) noexcept
       : _node(std::make_shared<node_type>(l, r)) {}

   /// Returns a shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _node.get()->evaluated(); }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_node.get()->node());
   }

   /// Evaluate a binary expression
   /// If the input expression has not been evaluated, it will evaluate it
   /// recursively before evaluating this expression
   [[maybe_unused]] auto eval() -> evaluated_type {
      return _node.get()->eval();
   }
};

} // namespace ten

#endif
