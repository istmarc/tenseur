#ifndef TENSEUR_EXPR_HXX
#define TENSEUR_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <Ten/Functional.hxx>
#include <Ten/Types.hxx>

namespace ten {

// FIXME This may be useful for getting Functional::Apply to work?
// Keep it here for now. F is the function type or Inner Function type
// Expression must accept a FunctionWrapper<...> or
// FunctionWrapper<..>::OuterFunctionWrapper<...>
template <template <class...> class F, class... Args> struct FunctionWrapper {
   template <template <template <class...> class /*F*/, class... /*Args*/>
             class G>
   struct OuterFunctionWrapper {};
};

namespace details {
// Node shape
template <class> struct NodeWrapper;

template <class T> struct NodeWrapper<ScalarNode<T>> {
   static auto ptr(const std::shared_ptr<ScalarNode<T>> &node) {
      return node.get();
   }
};

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct NodeWrapper<TensorNode<T, Shape, Order, Storage, Allocator>> {
   static auto
   shape(const std::shared_ptr<TensorNode<T, Shape, Order, Storage, Allocator>>
             &node) {
      return node.get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<TensorNode<T, Shape, Order, Storage, Allocator>>
           &node) {
      return node.get();
   }
};

template <class Input, class Output, template <typename...> class F,
          typename... Args>
struct NodeWrapper<UnaryNode<Input, Output, F, Args...>> {
   static auto
   shape(const std::shared_ptr<UnaryNode<Input, Output, F, Args...>> &node) {
      return node.get()->node().get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<UnaryNode<Input, Output, F, Args...>> &node) {
      return node.get()->node().get();
   }
};

template <class L, class R, class O, template <typename...> class F,
          typename... Args>
struct NodeWrapper<BinaryNode<L, R, O, F, Args...>> {
   static auto
   shape(const std::shared_ptr<BinaryNode<L, R, O, F, Args...>> &node) {
      return node.get()->node().get()->shape();
   }
   static auto
   ptr(const std::shared_ptr<BinaryNode<L, R, O, F, Args...>> &node) {
      return node.get()->node().get();
   }
};
} // namespace details

namespace details {
// Node type
template <class> struct OutputNodeType;

template <class T> struct OutputNodeType<ScalarNode<T>> {
   using type = ScalarNode<T>;
};

template <class T, class Shape, StorageOrder Order, class Storage,
          class Allocator>
struct OutputNodeType<TensorNode<T, Shape, Order, Storage, Allocator>> {
   using type = TensorNode<T, Shape, Order, Storage, Allocator>;
};

template <class Input, class Output, template <typename...> class F,
          class... Args>
struct OutputNodeType<::ten::UnaryNode<Input, Output, F, Args...>> {
   using type =
       typename ::ten::UnaryNode<Input, Output, F, Args...>::output_node_type;
};

template <class L, class R, class Output, template <typename...> class F,
          class... Args>
struct OutputNodeType<::ten::BinaryNode<L, R, Output, F, Args...>> {
   using type =
       typename ::ten::BinaryNode<L, R, Output, F, Args...>::output_node_type;
};
} // namespace details

// \class UnaryNode
// Apply a function to a ten::Tensor or a ten::Scalar
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
class UnaryNode {
 public:
   using input_node_type = typename details::OutputNodeType<Input>::type;
   using output_node_type = Output;
   using func_type = Func<input_node_type, Output>;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<isScalarNode<Output>::value,
                                             typename Output::scalar_type,
                                             typename Output::tensor_type>;

 private:
   /// Optional function type
   /// TODO remove optional
   std::optional<func_type> _func = std::nullopt;
   //// Output value
   std::shared_ptr<Output> _value = nullptr;
   std::shared_ptr<Input> _input = nullptr;

 public:
   UnaryNode() {}

   /// Construct a unary node if the function doesn't take additional parameters
   UnaryNode(const std::shared_ptr<Input> &input) noexcept
      requires(!::ten::functional::HasParams<func_type>::value)
       : _input(input), _func(func_type()) {}

   /// Construct a unary node if the function take additional parameters.
   /// The parameters of the functions args of type FuncArgs are forwarded
   /// to the constructor of the function when necessary.
   template <typename... FuncArgs>
   UnaryNode(const std::shared_ptr<Input> &input, FuncArgs... args) noexcept
      requires(::ten::functional::HasParams<func_type>::value)
       : _input(input), _func(func_type(std::forward<FuncArgs>(args)...)) {}

   [[nodiscard]] inline std::shared_ptr<Output> node() { return _value; }

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

      // Evaluate Input
      if constexpr (::ten::isUnaryNode<Input>::value ||
                    ::ten::isBinaryNode<Input>::value) {
         if (!_input.get()->evaluated()) {
            _input.get()->eval();
         }
      }

      // Allocate output
      if constexpr (::ten::isScalarNode<Output>::value) {
         _value.reset(new Output());
      } else if constexpr (!::ten::isScalarNode<Output>::value &&
                           Output::isStatic()) {
         _value.reset(new Output());
      } else {
         _value.reset(new Output(
             //::ten::details::NodeWrapper<Input>::shape(_input)
             //_func.value().outputShape(_input.get()->shape())
            _func.value().outputShape(::ten::details::NodeWrapper<Input>::shape(_input))
            ));
      }

      // Evaluate
      // if constexpr (::ten::functional::HasParams<func_type>::value) {
      _func.value()(*::ten::details::NodeWrapper<Input>::ptr(_input),
                    *_value.get());
      /*} else {
         func_type::operator()(*::ten::details::NodeWrapper<Input>::ptr(_input),
                               *_value.get());
      }*/

      return evaluated_type(_value);
   }
};

/// \class UnaryExpr
/// Unary expression.
template <typename E, template <class...> class Func, typename... Args>
class UnaryExpr : ::ten::Expr<UnaryExpr<E, Func, Args...>> {
 public:
   /// \typedef input_type
   /// Type of the input type of the function
   using input_node_type = typename ::ten::details::OutputNodeType<E>::type;

   /// \typedef output_type
   /// Type of the output type of the function
   using output_node_type = typename Func<input_node_type>::output_type;

   /// \typedef node_type
   /// Type of the node of the unary expresion
   using node_type = UnaryNode<E, output_node_type, Func, Args...>;

   /// \typedef expr_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<isScalarNode<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

   using func_type = Func<input_node_type>;

 private:
   /// Shared ptr to the node
   std::shared_ptr<node_type> _node;

 public:
   /// Construct a ten::UnaryNode from an expression
   explicit UnaryExpr(const std::shared_ptr<E> &expr) noexcept
      requires(!::ten::functional::HasParams<func_type>::value)
       : _node(std::make_shared<node_type>(expr)) {}

   template <typename... FuncArgs>
   explicit UnaryExpr(const std::shared_ptr<E> &expr,
                      FuncArgs &&...args) noexcept
      requires(::ten::functional::HasParams<func_type>::value)
       : _node(std::make_shared<node_type>(expr,
                                           std::forward<FuncArgs>(args)...)) {}

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

// \class BinaryNode
// Node of a binary expresion
// Left and Right can be ScalarNode, TensorNode or BinaryNode
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
class BinaryNode {
 public:
   /// Left input type
   using left_node_type = typename details::OutputNodeType<Left>::type;

   /// Right input type
   using right_node_type = typename details::OutputNodeType<Right>::type;

   /// Output type
   using output_node_type = Output;

   using func_type = Func<left_node_type, right_node_type, Output, Args...>;
   // using shape_type = typename Output::shape_type;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type = std::conditional_t<isScalarNode<Output>::value,
                                             typename Output::scalar_type,
                                             typename Output::tensor_type>;

 private:
   std::optional<func_type> _func = std::nullopt;
   // std::optional<typename Output::shape_type> _shape = std::nullopt;
   std::shared_ptr<Output> _value = nullptr;
   std::shared_ptr<Left> _left;
   std::shared_ptr<Right> _right;

 public:
   BinaryNode() {}

   BinaryNode(const std::shared_ptr<Left> &left,
              const std::shared_ptr<Right> &right) noexcept
      requires(!::ten::functional::HasParams<func_type>::value)
       : _left(left), _right(right), _func(func_type()) {}

   template <typename... FuncArgs>
   BinaryNode(const std::shared_ptr<Left> &left,
              const std::shared_ptr<Right> &right, FuncArgs... args) noexcept
      requires(::ten::functional::HasParams<func_type>::value)
       : _func(func_type(std::forward<FuncArgs>(args)...)), _left(left),
         _right(right) {}

   [[nodiscard]] inline std::shared_ptr<Output> node() { return _value; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _value.get(); }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_value);
   }

   [[maybe_unused]] auto eval() noexcept -> evaluated_type {
      if (_value)
         return evaluated_type(_value);

      // Evaluate LeftExpr
      if constexpr (::ten::isUnaryNode<Left>::value ||
                    ::ten::isBinaryNode<Left>::value) {
         if (!_left.get()->evaluated()) {
            _left.get()->eval();
         }
      }
      // Evaluate RightExpr
      if constexpr (::ten::isUnaryNode<Right>::value ||
                    ::ten::isBinaryNode<Right>::value) {
         if (!_right.get()->evaluated()) {
            _right.get()->eval();
         }
      }

      if constexpr (Output::isStatic()) {
         _value.reset(new Output());
      } else {
         // func(TensorNode, TensorNode)
         if constexpr (!::ten::isScalarNode<Left>::value &&
                       !::ten::isScalarNode<Right>::value) {
            if (::ten::Debug)
               std::cout << "f(Tensor, Tensor)" << std::endl;
            _value.reset(new Output(func_type::outputShape(
                ::ten::details::NodeWrapper<Left>::shape(_left),
                ::ten::details::NodeWrapper<Right>::shape(_right))));
         } else {
            // func(ScalarNode, TensorNode)
            if constexpr (::ten::isScalarNode<Left>::value &&
                          !::ten::isScalarNode<Right>::value) {

               if (::ten::Debug)
                  std::cout << "f(Scalar, Tensor)" << std::endl;
               _value.reset(new Output(func_type::outputShape(
                   ::ten::details::NodeWrapper<Right>::shape(_right))));
            }
            // func(TensorNode, ScalarNode)
            if constexpr (!::ten::isScalarNode<Left>::value &&
                          ::ten::isScalarNode<Right>::value) {
               if (::ten::Debug)
                  std::cout << "f(Tensor, Scalar)" << std::endl;
               _value.reset(new Output(func_type::outputShape(
                   ::ten::details::NodeWrapper<Left>::shape(_left))));
            }
         }
      }

      // Call F
      _func.value()(*details::NodeWrapper<Left>::ptr(_left),
                    *details::NodeWrapper<Right>::ptr(_right), *_value.get());
      /*func_type::operator()(*details::NodeWrapper<Left>::ptr(_left),
       *details::NodeWrapper<Right>::ptr(_right),
       *_value.get());
       */

      return evaluated_type(_value);
   }
};

/// \class BinaryExpr
/// Binary expression
// Left and Right can be ScalarNode, TensorNode or BinaryExpr
// left is std::shared_ptr<Left> and right is std::shared_ptr<Right>
template <typename Left, typename Right, template <typename...> class Func,
          typename... Args>
class BinaryExpr : ::ten::Expr<BinaryExpr<Left, Right, Func, Args...>> {
 public:
   /// Left input type
   using left_node_type = typename details::OutputNodeType<Left>::type;

   /// Right input type
   using right_node_type = typename details::OutputNodeType<Right>::type;

   /// output_node_type is ScalarNode or TensorNode
   using output_node_type =
       typename Func<left_node_type, right_node_type>::output_type;

   // Node type
   using node_type = BinaryNode<Left, Right, output_node_type, Func, Args...>;

   /// \typedef evaluated_type
   /// Type of the evaluated expression
   using evaluated_type =
       std::conditional_t<isScalarNode<output_node_type>::value,
                          typename output_node_type::scalar_type,
                          typename output_node_type::tensor_type>;

 private:
   std::shared_ptr<node_type> _node;

 public:
   /// Construct a BinaryExpr from an expression
   explicit BinaryExpr(const std::shared_ptr<Left> &left,
                       const std::shared_ptr<Right> &right) noexcept
       : _node(std::make_shared<node_type>(left, right)) {}

   /// Returns a shared pointer to the node of the expression
   [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

   /// Returns whether the expression is evaluated
   [[nodiscard]] bool evaluated() const { return _node.get()->evaluated(); }

   /// Returns the the evaluated expression of type ten::Scalar of ten::Tensor
   [[nodiscard]] auto value() -> evaluated_type {
      return evaluated_type(_node.get()->node());
   }

   /// Evaluate a binary expression and return ten::ScalarNode or
   /// ten::TensorNode If the input expression has not been evaluated, it will
   /// evaluate it recursively before evaluating this expression
   [[maybe_unused]] auto eval() -> evaluated_type {
      return _node.get()->eval();
   }
};

} // namespace ten

#endif
