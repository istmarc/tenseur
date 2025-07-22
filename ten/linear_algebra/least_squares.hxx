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
   explicit linear_system(ls_options &&options)
       : _options(std::move(options)) {}

   /// Solve Ax=b
   void solve(const matrix<T> &A, const vector<T> &b) {
      if (_options._method == ls_method::qr) {
         ::ten::linalg::qr<T> qr_fact;
         qr_fact.factorize(A);
         auto [q, r] = qr_fact.factors();
         ::ten::vector<T> z = ::ten::transposed(q) * b;
         size_t n = b.size();
         _x = ten::vector<T>({n});
         ::ten::linalg::backward_subtitution(r, z, _x);
      } else if (_options._method == ls_method::lu) {
         ::ten::linalg::lu<T> lu_fact;
         lu_fact.factorize(A);
         auto [P, L, U] = lu_fact.factors();
         // Solve Lz = t using forward subtitution where z = Ux and t = P^T b
         ::ten::vector<T> t = ::ten::transposed(P) * b;
         size_t n = b.size();
         auto z = ::ten::vector<T>({n});
         ::ten::linalg::forward_subtitution(L, t, z);
         // Solve Ux = z using backward subtitution
         _x = ten::vector<T>({n});
         ::ten::linalg::backward_subtitution(U, z, _x);
      } else if (_options._method == ls_method::svd) {
         ::ten::linalg::svd<T> svd_fact;
         svd_fact.factorize(A);
         auto [U, Sigma, Vt] = svd_fact.factors();
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

/// TODO Linear least squares

/// TODO Nonlinear least squares

} // namespace ten::linalg

#endif
