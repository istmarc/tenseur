#ifndef TENSEUR_KERNELS_INPLACE_FUNCTIONS
#define TENSEUR_KERNELS_INPLACE_FUNCTIONS

#include <ten/types.hxx>

namespace ten::kernels::inplace{

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void cum_sum(X& x) {
   for (size_t i = 1; i < x.size(); i++) {
      x[i] += x[i-1];
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void abs(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::abs(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void sqrt(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sqrt(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void sqr(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = x[i] * x[i];
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void sin(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sin(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void sinh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::sinh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void asin(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::asin(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void cos(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::cos(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void cosh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::cosh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void acos(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::acos(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void tan(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::tan(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void tanh(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::tanh(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void atan(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::atan(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void exp(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::exp(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void log(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::log(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void log10(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::log10(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void floor(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::floor(x[i]);
   }
};

template<class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void ceil(X& x) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::ceil(x[i]);
   }
};

template<class T, class X>
requires(::ten::is_tensor<X>::value || ::ten::is_column<X>::value || ::ten::is_row<X>::value)
void pow(X& x, const T n) {
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = std::pow(x[i], n);
   }
};

}

#endif
