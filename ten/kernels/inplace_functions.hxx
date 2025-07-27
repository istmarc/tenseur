#ifndef TENSEUR_KERNELS_INPLACE_FUNCTIONS
#define TENSEUR_KERNELS_INPLACE_FUNCTIONS

#include <ten/types.hxx>

namespace ten::kernels{

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_cum_sum(X& x) {
   for (size_t i = 1; i < x.size(); i++) {
      x[i] += x[i-1];
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_abs(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::abs(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_sqrt(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sqrt(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_sqr(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = x[i] * x[i];
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_sin(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sin(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_sinh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sinh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_asin(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::asin(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_cos(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::cos(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_cosh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::cosh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_acos(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::acos(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_tan(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::tan(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_tanh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::tanh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_atan(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::atan(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_exp(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::exp(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_log(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::log(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_log10(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::log10(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_floor(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::floor(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_ceil(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::ceil(x[i]);
   }
};

template<class T, class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void inplace_pow(X& x, const T n) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::pow(x[i], n);
   }
};

}

#endif
