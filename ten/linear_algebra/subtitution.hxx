#ifndef TENSUER_LINEAR_ALGEBRA_SUBTITUTION
#define TENSUER_LINEAR_ALGEBRA_SUBTITUTION

#include <ten/tensor>

namespace ten::linalg {

template <class M, class V>
   requires(::ten::is_matrix<M>::value && ::ten::is_vector<V>::value)
void forward_subtitution(const M &L, const V &y, V &x) {
   using T = typename V::value_type;
   size_t n = x.size();
   x[0] = y[0] / L(0, 0);
   for (size_t i = 1; i < n; i++) {
      T s = y[i];
      for (size_t j = 0; j < i; j++) {
         s -= L(i, j) * x[j];
      }
      x[i] = s / L(i, i);
   }
}

template <class M, class V>
   requires(::ten::is_matrix<M>::value && ::ten::is_vector<V>::value)
void backward_subtitution(const M &U, const V &y, V &x) {
   using T = typename V::value_type;
   long n = static_cast<long>(x.size());
   x[n - 1] = y[n - 1] / U(n - 1, n - 1);
   for (long i = n - 2; i >= 0; i--) {
      T s = y[i];
      for (long j = i + 1; j < n; j++) {
         s -= U(i, j) * x[j];
      }
      x[i] = s / U(i, i);
   }
}

} // namespace ten::linalg

#endif
