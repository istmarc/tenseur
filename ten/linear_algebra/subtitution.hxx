#ifndef TENSUER_LINEAR_ALGEBRA_SUBTITUTION
#define TENSUER_LINEAR_ALGEBRA_SUBTITUTION

#include <ten/tensor>

namespace ten {

namespace linalg {

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

} // namespace linalg

} // namespace ten

#endif
