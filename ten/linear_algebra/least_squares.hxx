#ifndef TRENCH_LEASTSQUARES_LINEAR
#define TRENCH_LEASTSQUARES_LINEAR

#include <ten/linear_algebra/factorization.hxx>
#include <ten/linear_algebra/subtitution.hxx>
#include <ten/types.hxx>
#include <type_traits>

namespace ten {
namespace linalg {

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
      size_t n = b.size();
      _x = ten::vector<T>({n});

      if (_options._method == ls_method::qr) {
         ::ten::qr<T> qr_fact;
         qr_fact.factorize(A);
         auto [q, r] = qr_fact.factors();
         ten::matrix<T> qt = ten::transpose(q);
         ten::vector<T> z = qt * b;
         ten::linalg::backward_subtitution(r, z, _x);
      }
      /* else if (_options.method == ls_method::lu) {
         ::ten::lu<T> lu_fact;
         lu_fact.factorize(A);
         auto [l, u] = lu_fact.factors();
      } else if (_options.method == ls_method::svd) {
         ::ten::svd<T> svd_fact;
         svd_fact.factorize(A);
         auto [s, v, d] = svd_fact.factors();
      }*/
   }

   ::ten::vector<T> x() { return _x; }
};

/// TODO Linear least squares

/// TODO Nonlinear least squares

} // namespace linalg

} // namespace ten

#endif
