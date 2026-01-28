#ifndef TRENCH_LEASTSQUARES_LINEAR
#define TRENCH_LEASTSQUARES_LINEAR

#include <ten/linear_algebra/factorization.hxx>
#include <ten/linear_algebra/subtitution.hxx>
#include <ten/types.hxx>
#include <type_traits>

namespace ten::linalg {

enum class ls_method { qr = 1, lu = 2, svd = 3 };

struct ls_options {
   ls_method _method = ls_method::qr;

   ls_options(ls_method method) : _method(method) {}
};

/// Linear system
template <class T = float> class linear_system {
   static_assert(std::is_floating_point_v<T>, "T must be floating point");

 private:
   vector<T> _x;
   ls_options _options;

 public:
   explicit linear_system(const ls_options &options)
       : _options(options) {}

   /// Solve Ax=b
   void solve(matrix<T> &A, vector<T> &b) {
      if (_options._method == ls_method::qr) {
         auto [q, r] = ::ten::linalg::qr(A);
         ::ten::vector<T> z = ::ten::transposed(q) * b;
         size_t n = b.size();
         _x = ten::vector<T>({n});
         ::ten::linalg::backward_subtitution(r, z, _x);
      } else if (_options._method == ls_method::lu) {
         auto [P, L, U] = ::ten::linalg::lu(A);
         // Solve Lz = t using forward subtitution where z = Ux and t = P^T b
         ::ten::vector<T> t = ::ten::transposed(P) * b;
         size_t n = b.size();
         auto z = ::ten::vector<T>({n});
         ::ten::linalg::forward_subtitution(L, t, z);
         // Solve Ux = z using backward subtitution
         _x = ten::vector<T>({n});
         ::ten::linalg::backward_subtitution(U, z, _x);
      } else if (_options._method == ls_method::svd) {
         auto [U, Sigma, Vt] = ::ten::linalg::svd(A);
         size_t n = b.size();
         // TODO Make invSigma = ten::fill<diagonal<T>>({n, n}, 1) / Sigma
         ::ten::diagonal<T> invSigma({n, n});
         for (size_t i = 0; i < n; i++) {
            invSigma[i] = T(1) / Sigma[i];
         }
         // FIXME Make this work _x = ::ten::transposed(Vt) *
         // ::ten::dense(invSigma) * ::ten::transposed(U) * b;
         ::ten::matrix<T> m = ::ten::transposed(Vt) * ::ten::dense(invSigma) *
                              ::ten::transposed(U);
         _x = m * b;
      }
   }

   ::ten::vector<T> solution() { return _x; }
};

/// Solve Ax=b
template<Matrix M, Vector V>
requires(std::is_same_v<typename M::value_type, typename V::value_type>)
auto solve(M&& A, V&& b, const ls_method method = ls_method::qr) -> decltype(auto) {
   using value_type = M::value_type;
   ls_options options(method);
   ::ten::linalg::linear_system<value_type> ls(options);
   ls.solve(A, b);
   return ls.solution();
}

/// TODO Linear least squares

/// TODO Nonlinear least squares

} // namespace ten::linalg

#endif
