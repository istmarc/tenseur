#ifndef TENSEUR_IOS_EXPR
#define TENSEUR_IOS_EXPR

#include <iostream>
#include <ten/types.hxx>
#include <ten/utils.hxx>

namespace ten {

/// print unary_expr
template <class Input, class Output, template <typename...> class Func,
          typename... Args>
std::ostream &
operator<<(std::ostream &os,
           ::ten::unary_expr<Input, Output, Func, Args...> &expr) {
   using expr_type = ::ten::unary_expr<Input, Output, Func, Args...>;
   if (expr.evaluated()) {
      os << "evaluated ";
   } else {
      os << "unevaluated ";
   }
   os << "unary_expr(Input: ";
   if constexpr (::ten::is_tensor_v<Input>) {
      os << "tensor<";
   }
   if constexpr (::ten::is_column_v<Input>) {
      os << "column<";
   }
   if constexpr (::ten::is_row_v<Input>) {
      os << "row<";
   }
   if constexpr (::ten::is_tensor_v<Input> || ::ten::is_column_v<Input> ||
                 ::ten::is_row_v<Input>) {
      os << ::ten::to_string<typename Input::value_type>();
      os << ">";
   }
   if constexpr (::ten::is_unary_expr<Input>::value) {
      os << expr.input();
   }
   os << ", Output: ";
   if constexpr (::ten::is_scalar<Output>::value) {
      os << "scalar<";
   }
   if constexpr (::ten::is_tensor_v<Output>) {
      os << "tensor<";
   }
   if constexpr (::ten::is_column_v<Output>) {
      os << "column<";
   }
   if constexpr (::ten::is_row_v<Output>) {
      os << "row<";
   }
   os << ::ten::to_string<typename Output::value_type>();
   os << ">, Function: ";
   os << expr_type::func_type::name();
   os << ")";
   return os;
}

/// Print binary_expr
template <class Left, class Right, class Output,
          template <typename...> class Func, typename... Args>
std::ostream &
operator<<(std::ostream &os,
           ::ten::binary_expr<Left, Right, Output, Func, Args...> &expr) {
   using expr_type = ::ten::binary_expr<Left, Right, Output, Func, Args...>;
   if (expr.evaluated()) {
      os << "evaluated ";
   } else {
      os << "unevaluated ";
   }
   // Print the left input
   os << "binary_expr(Left: ";
   if constexpr (::ten::is_tensor_v<Left>) {
      os << "tensor<";
   }
   if constexpr (::ten::is_column_v<Left>) {
      os << "column<";
   }
   if constexpr (::ten::is_row_v<Left>) {
      os << "row<";
   }
   if constexpr (::ten::is_tensor_v<Left> || ::ten::is_column_v<Left> ||
                 ::ten::is_row_v<Left>) {
      os << ::ten::to_string<typename Left::value_type>();
      os << ">";
   }
   if constexpr (::ten::is_unary_expr<Left>::value ||
                 ::ten::is_binary_expr<Left>::value) {
      if constexpr (::ten::is_unary_expr<Left>::value) {
         os << "unary_expr";
      } else {
         os << "binary_expr";
         ///os << expr.left();
      }
   }
   // print the right input
   os << ", Right: ";
   if constexpr (::ten::is_tensor_v<Right>) {
      os << "tensor<";
   }
   if constexpr (::ten::is_column_v<Right>) {
      os << "column<";
   }
   if constexpr (::ten::is_row_v<Right>) {
      os << "row<";
   }
   if constexpr (::ten::is_tensor_v<Right> || ::ten::is_column_v<Right> ||
                 ::ten::is_row_v<Right>) {
      os << ::ten::to_string<typename Right::value_type>();
      os << ">";
   }
   if constexpr (::ten::is_unary_expr<Right>::value ||
                 ::ten::is_binary_expr<Right>::value) {
      if constexpr (::ten::is_unary_expr<Left>::value) {
         os << "unary_expr";
      } else {
         os << "binary_expr";
         ///os << expr.left();
      }
      //os << expr.right();
   }
   // Print the output
   os << ", Output: ";
   if constexpr (::ten::is_scalar<Output>::value) {
      os << "scalar<";
   }
   if constexpr (::ten::is_tensor_v<Output>) {
      os << "tensor<";
   }
   if constexpr (::ten::is_column_v<Output>) {
      os << "column<";
   }
   if constexpr (::ten::is_row_v<Output>) {
      os << "row<";
   }
   os << ::ten::to_string<typename Output::value_type>();
   // Print the function
   os << ">, Function: ";
   os << expr_type::func_type::name();
   os << ")";
   return os;
}

} // namespace ten

#endif
