#ifndef TRENCH_LEASTSQUARES_LINEAR
#define TRENCH_LEASTSQUARES_LINEAR

#include <ten/types.hxx>
#include <type_traits>

namespace tr{

enum class ls_method{
   qr = 1,
   lu = 2,
   svd = 3
};

struct ls_options{
   ls_method method = ls_method::qr;
};

template<class __t = float>
class ls{
   static_assert(std::is_floating_point_v<__t>,
      "T must be floating point");
private:
   matrix<__t> _A;
   vector<__t> _b;
   vector<__t> _x;
   ls_options _options;

public:
   explicit ls(ls_options options) : _options(options) {}

   /// Solve Ax=b
   void solve(const matrix<__t>& A, const vector<__t>& b) {
   }
};

/// TODO Nonlinear least squares

};

#endif
