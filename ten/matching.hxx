#ifndef TENSEUR_MATCHING
#define TENSEUR_MATCHING

#include <ten/functional_types.hxx>
#include <ten/functions.hxx>
#include <ten/types.hxx>

#include <iostream>
#include <type_traits>

namespace ten {

////////////////////////////////////////////////////////////////////////////////
/// Match gemm for dense matrix
/// C = A*B + C
/// or C = binary_expr(left: binary_expr(left: A, right: B, func:mul), right: C,
/// func: add)
template <Expr ExprType, Matrix C>
bool match_gemm(ExprType &&expr, const C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_binary_expr_v<expr_type>) {
      return false;
   } else {
      using left_type = expr_type::left_ty;
      // Left type must be binary expr (A, B, mul)
      if constexpr (!::ten::is_binary_expr<left_type>::value) {
         return false;
      } else {
         // Function must be mul
         // left and right of left can be matrices or expressions
         using left_func_type = left_type::func_type;
         // left_func_type
         if (!::ten::is_mul<left_func_type>::value) {
            return false;
         }
         // Right type must be a matrix or static matrix, equal to c
         using right_type = expr_type::right_ty;
         if constexpr (!::ten::is_matrix<right_type>::value &&
                       !::ten::is_smatrix<right_type>::value) {
            return false;
         } else {
            // Right type and output (x) must be of the same type
            if (!std::is_same_v<right_type, C>) {
               return false;
            }
            // They must also be equal
            auto right = expr.right();
            if (right != x) {
               return false;
            }
         }
         // The function must be elementwise add
         using func_type = expr_type::func_type;
         return ::ten::is_add<func_type>::value;
      }
   }
}

template <Expr ExprType, Matrix C>
// FIXME Currently GEMM support only float and double types
   requires(::ten::is_float<typename C::value_type>::value ||
            ::ten::is_double<typename C::value_type>::value)
void fuse_gemm(ExprType &&expr, C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (ten::is_expr<expr_type>::value) {
      auto left = expr.left();
      if constexpr (ten::is_expr<decltype(left)>::value) {
         auto A = left.left();
         auto B = left.right();
         using T = C::value_type;
         ten::gemm(T(1), A, B, T(1), x);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Match GEMM C <- alpha * A * B + C
// C = binary_expr(binary_expr(binary_expr(alpha, A), B), func:mul), C,
// func:add)
template <Expr ExprType, Matrix C>
bool match_gemm_alpha_nobeta(ExprType &&expr, const C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_binary_expr_v<expr_type>) {
      return false;
   } else {
      using left_type = expr_type::left_ty;
      // Left type must be binary expr (binary_expr(alpha, A), B, mul)
      if constexpr (!::ten::is_binary_expr_v<left_type>) {
         return false;
      } else {
         // Match scalar * matrix
         using left_left_type = left_type::left_ty;
         if constexpr (!::ten::is_binary_expr_v<left_left_type>) {
            return false;
         } else {
            // Function must be mul
            // left and right of left can be matrices or expressions
            using left_left_func_type = left_left_type::func_type;
            // left_func_type must be elementwise multiplication
            if (!::ten::is_mul<left_left_func_type>::value) {
               return false;
            }
            // left of left_left must be a scalar
            using left_left_left = left_left_type::left_ty;
            if (!::ten::is_scalar<left_left_left>::value) {
               return false;
            }
            // Right of left_left must be a matrix
            using right_left_left = left_left_type::right_ty;
            if (!::ten::is_matrix_v<right_left_left> &&
                !::ten::is_smatrix_v<right_left_left>) {
               return false;
            }
         }
      }
      // Right of left must be a matrix
      using right_left_type = left_type::right_ty;
      if constexpr (!::ten::is_matrix_v<right_left_type> &&
                    !::ten::is_smatrix_v<right_left_type>) {
         return false;
      }
      // Right type must be a matrix or static matrix, equal to c
      using right_type = expr_type::right_ty;
      if constexpr (!::ten::is_matrix<right_type>::value &&
                    !::ten::is_smatrix<right_type>::value) {
         return false;
      } else {
         // Right type and output (x) must be of the same type
         if (!std::is_same_v<right_type, C>) {
            return false;
         }
         // They must also be equal
         auto right = expr.right();
         if (right != x) {
            return false;
         }
      }
      // The function must be elementwise add
      using func_type = expr_type::func_type;
      return ::ten::is_add<func_type>::value;
   }
}

template <Expr ExprType, Matrix C>
// FIXME Currently GEMM support only float and double types
   requires(::ten::is_float<typename C::value_type>::value ||
            ::ten::is_double<typename C::value_type>::value)
void fuse_gemm_alpha_nobeta(ExprType &&expr, C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (ten::is_expr<expr_type>::value) {
      auto left = expr.left();
      if constexpr (ten::is_binary_expr<decltype(left)>::value) {
         auto left_left = left.left();
         if constexpr (ten::is_binary_expr<decltype(left_left)>::value) {
            auto alpha = left_left.left();
            if constexpr (ten::is_scalar<decltype(alpha)>::value) {
               auto A = left_left.right();
               auto B = left.right();
               using T = C::value_type;
               ten::gemm(alpha.value(), A, B, T(1), x);
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Match GEMM C <- A * B + beta * C
template <Expr ExprType, Matrix C>
bool match_gemm_noalpha_beta(ExprType &&expr, const C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_binary_expr_v<expr_type>) {
      return false;
   } else {
      using left_type = expr_type::left_ty;
      // Left type must be binary expr (binary_expr(alpha, A), B, mul)
      if constexpr (!::ten::is_binary_expr_v<left_type>) {
         return false;
      } else {
         // Match matrix * matrix
         // Function must be mul
         using left_func_type = left_type::func_type;
         // left_func_type must be elementwise multiplication
         if (!::ten::is_mul<left_func_type>::value) {
            return false;
         }
         // Right of left_left must be a matrix
         using left_left = left_type::left_ty;
         if (!::ten::is_matrix_v<left_left> &&
             !::ten::is_smatrix_v<left_left>) {
            return false;
         }
         // Right of left_left must be a matrix
         using right_left = left_type::right_ty;
         if (!::ten::is_matrix_v<right_left> &&
             !::ten::is_smatrix_v<right_left>) {
            return false;
         }
      }
      // Right type must be an expression binary_expr(beta, C)
      using right_type = expr_type::right_ty;
      if constexpr (!::ten::is_binary_expr_v<right_type>) {
         return false;
      } else {
         // Left or right type must be an scalar
         using left_right = right_type::left_ty;
         if (!ten::is_scalar<left_right>::value) {
            return false;
         }
         // right of right type must be a matrix
         using right_right_type = right_type::right_ty;
         if (!::ten::is_matrix<right_right_type>::value &&
             !::ten::is_smatrix<right_right_type>::value) {
            return false;
         }
         // Right type and output (x) must be of the same type
         if (!std::is_same_v<right_right_type, C>) {
            return false;
         }
         // They must also be equal
         auto right_right = expr.right().right();
         if (right_right != x) {
            return false;
         }
         // The function of right must be elementwise mul
         using right_func_type = right_type::func_type;
         if (!::ten::is_mul<right_func_type>::value) {
            return false;
         }
      }
      // The function must be elementwise add
      using func_type = expr_type::func_type;
      return ::ten::is_add<func_type>::value;
   }
}

template <Expr ExprType, Matrix C>
// FIXME Currently GEMM support only float and double types
   requires(::ten::is_float<typename C::value_type>::value ||
            ::ten::is_double<typename C::value_type>::value)
void fuse_gemm_noalpha_beta(ExprType &&expr, C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   using T = typename C::value_type;
   if constexpr (ten::is_expr<expr_type>::value) {
      auto left = expr.left();
      if constexpr (ten::is_binary_expr<decltype(left)>::value) {
         auto right = expr.right();
         if constexpr (ten::is_binary_expr_v<decltype(right)>) {
            auto beta = right.left();
            if constexpr (ten::is_scalar<decltype(beta)>::value) {
               auto A = left.left();
               auto B = left.right();
               using T = C::value_type;
               ten::gemm(T(1), A, B, beta.value(), x);
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Match GEMM C <- alpha * A * B + beta * C
template <Expr ExprType, Matrix C>
bool match_gemm_alpha_beta(ExprType &&expr, const C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_binary_expr_v<expr_type>) {
      return false;
   } else {
      using left_type = expr_type::left_ty;
      // Left type must be binary expr (binary_expr(alpha, A), B, mul)
      if constexpr (!::ten::is_binary_expr_v<left_type>) {
         return false;
      } else {
         // Match scalar * matrix
         using left_left_type = left_type::left_ty;
         if constexpr (!::ten::is_binary_expr_v<left_left_type>) {
            return false;
         } else {
            // Function must be mul
            // left and right of left can be matrices or expressions
            using left_left_func_type = left_left_type::func_type;
            // left_func_type must be elementwise multiplication
            if (!::ten::is_mul<left_left_func_type>::value) {
               return false;
            }
            // left of left_left must be a scalar
            using left_left_left = left_left_type::left_ty;
            if (!::ten::is_scalar<left_left_left>::value) {
               return false;
            }
            // Right of left_left must be a matrix
            using right_left_left = left_left_type::right_ty;
            if (!::ten::is_matrix_v<right_left_left> &&
                !::ten::is_smatrix_v<right_left_left>) {
               return false;
            }
         }
      }
      // Right of left must be a matrix
      using right_left_type = left_type::right_ty;
      if constexpr (!::ten::is_matrix_v<right_left_type> &&
                    !::ten::is_smatrix_v<right_left_type>) {
         return false;
      }
      // Right type must be an expression binary_expr(beta, C)
      using right_type = expr_type::right_ty;
      if constexpr (!::ten::is_binary_expr_v<right_type>) {
         return false;
      } else {
         // Left or right type must be an scalar
         using left_right = right_type::left_ty;
         if (!ten::is_scalar<left_right>::value) {
            return false;
         }
         // right of right type must be a matrix
         using right_right_type = right_type::right_ty;
         if (!::ten::is_matrix<right_right_type>::value &&
             !::ten::is_smatrix<right_right_type>::value) {
            return false;
         }
         // Right type and output (x) must be of the same type
         if (!std::is_same_v<right_right_type, C>) {
            return false;
         }
         // They must also be equal
         auto right_right = expr.right().right();
         if (right_right != x) {
            return false;
         }
         // The function of right must be elementwise mul
         using right_func_type = right_type::func_type;
         if (!::ten::is_mul<right_func_type>::value) {
            return false;
         }
      }
      // The function must be elementwise add
      using func_type = expr_type::func_type;
      return ::ten::is_add<func_type>::value;
   }
}

template <Expr ExprType, Matrix C>
// FIXME Currently GEMM support only float and double types
   requires(::ten::is_float<typename C::value_type>::value ||
            ::ten::is_double<typename C::value_type>::value)
void fuse_gemm_alpha_beta(ExprType &&expr, C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (ten::is_expr<expr_type>::value) {
      auto left = expr.left();
      if constexpr (ten::is_binary_expr<decltype(left)>::value) {
         auto left_left = left.left();
         if constexpr (ten::is_binary_expr<decltype(left_left)>::value) {
            auto alpha = left_left.left();
            if constexpr (ten::is_scalar<decltype(alpha)>::value) {
               auto right = expr.right();
               if constexpr (ten::is_binary_expr_v<decltype(right)>) {
                  auto beta = right.left();
                  if constexpr (ten::is_scalar<decltype(beta)>::value) {
                     auto A = left_left.right();
                     auto B = left.right();
                     using T = C::value_type;
                     ten::gemm(alpha.value(), A, B, beta.value(), x);
                  }
               }
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Match GEMM C <- A * B
// or C = binary_expr(A, B, func: mul)

template <Expr ExprType, Matrix C>
bool match_gemm_noalpha_nobeta(ExprType &&expr, const C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_binary_expr_v<expr_type>) {
      return false;
   } else {
      using func_type = expr_type::func_type;
      // func_type must be mul
      return ::ten::is_mul<func_type>::value;
   }
}

template <Expr ExprType, Matrix C>
// FIXME Currently GEMM support only float and double types
   requires(::ten::is_float<typename C::value_type>::value ||
            ::ten::is_double<typename C::value_type>::value)
void fuse_gemm_noalpha_nobeta(ExprType &&expr, C &x) {
   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (ten::is_expr<expr_type>::value) {
      auto A = expr.left();
      auto B = expr.right();
      using T = C::value_type;
      ten::gemm(T(1), A, B, T(0), x);
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Match sqrt
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
bool match_sqrt(ExprType &&expr, const T &x) {

   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_unary_expr_v<expr_type>) {
      return false;
   } else {
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
}

/// Fuse sqrt inplace
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
void fuse_sqrt(ExprType &&expr, T &x) {
   ::ten::kernels::inplace::sqrt(x);
}

////////////////////////////////////////////////////////////////////////////////
/// Match sqr
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
bool match_sqr(ExprType &&expr, const T &x) {

   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_unary_expr_v<expr_type>) {
      return false;
   } else {
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
      return ::ten::is_sqr<func_type>::value;
   }
}

/// Fuse sqr inplace
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
void fuse_sqr(ExprType &&expr, T &x) {
   ::ten::kernels::inplace::sqr(x);
}

////////////////////////////////////////////////////////////////////////////////
/// Match abs
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
bool match_abs(ExprType &&expr, const T &x) {

   using expr_type = std::remove_cvref_t<ExprType>;
   if constexpr (!::ten::is_unary_expr_v<expr_type>) {
      return false;
   } else {
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
      return ::ten::is_abs<func_type>::value;
   }
}

/// Fuse abs inplace
template <Expr ExprType, class T>
   requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> ||
            ::ten::is_row_v<T>)
void fuse_abs(ExprType &&expr, T &x) {
   ::ten::kernels::inplace::abs(x);
}

////////////////////////////////////////////////////////////////////////////////
/// Match and fuse
template <Expr ExprType, class X>
   requires(::ten::is_tensor_v<X> || ::ten::is_column_v<X> ||
            ::ten::is_row_v<X>)
bool match_fuse(ExprType &&expr, X &x) {

   if constexpr (::ten::is_matrix_v<X> || ::ten::is_smatrix<X>::value) {
      if (match_gemm_noalpha_nobeta(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched gemm with no alpha, no beta\n";
         }
         fuse_gemm_noalpha_nobeta(expr, x);
         return true;
      }

      if (match_gemm_alpha_nobeta(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched gemm with alpha, no beta\n";
         }
         fuse_gemm_alpha_nobeta(expr, x);
         return true;
      }

      if (match_gemm_noalpha_beta(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched gemm with no alpha, beta\n";
         }
         fuse_gemm_noalpha_beta(expr, x);
         return true;
      }

      if (match_gemm_alpha_beta(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched gemm with alpha, beta\n";
         }
         fuse_gemm_alpha_beta(expr, x);
         return true;
      }

      if (match_gemm(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched gemm\n";
         }
         fuse_gemm(expr, x);
         return true;
      }
      // TODO Match gemv
   }

   if constexpr (::ten::is_vector_v<X> || ten::is_column_v<X> ||
                 ten::is_row_v<X>) {
      if (match_sqrt(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched inplace sqrt\n";
         }
         fuse_sqrt(expr, x);
         return true;
      }

      if (match_sqr(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched inplace sqr\n";
         }
         fuse_sqr(expr, x);
         return true;
      }

      if (match_abs(expr, x)) {
         if (::ten::is_verbose) {
            std::cout << "Matched inplace abs\n";
         }
         fuse_abs(expr, x);
         return true;
      }
   }

   if (::ten::is_verbose) {
      std::cout << "The expression wasn't matched and fused\n";
   }
   return false;
}

} // namespace ten

#endif
