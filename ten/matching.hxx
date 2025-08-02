#ifndef TENSEUR_MATCHING
#define TENSEUR_MATCHING

#include <ten/functional_types.hxx>
#include <ten/types.hxx>

#include <iostream>

namespace ten {

/// Match gemm for dense matrix
/// m = a*b + m
/// m = add()
/*template<Expr ExprType, Matrix A>
bool match_gemm(ExprType&& expr, A& m) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if (!::ten::is_binary_expr<expr_type>::value) {
      return false;
   }
   auto left = expr.left();
   auto right = expr.right();
   if () {
   }
}*/

/// Match sqrt
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
bool match_sqrt(ExprType &&expr, const T &x) {

   using expr_type = std::remove_cvref_t<ExprType>;
   if (!::ten::is_unary_expr<expr_type>::value) {
      return false;
   }
   using input_type = expr_type::input_ty;
   // Input type must be tensor or column or row
   if (!::ten::is_tensor_v<T> && !::ten::is_column_v<T> &&
       !::ten::is_row_v<T>) {
      return false;
   }
   auto input = expr.input();
   // Input and output (x) must be of the same type
   if (!std::is_same_v<input_type, T>) {
      return false;
   }
   // And must be equal
   if (input != x) {
      return false;
   }
   using func_type = expr_type::func_type;
   return ::ten::is_sqrt<func_type>::value;
}

/// Fuse sqrt inplace
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
void fuse_sqrt(ExprType &&expr, T &x) {
   ::ten::kernels::inplace_sqrt(x);
}

/// Macth and fuse
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
bool match_fuse(ExprType &&expr, T &x) {
   // TODO Match gemm
   // TODO Match gemv
   if (match_sqrt(expr, x)) {
      fuse_sqrt(expr, x);
      return true;
   }
   // The expression wasn't matched and fused
   return false;
}

} // namespace ten

#endif
