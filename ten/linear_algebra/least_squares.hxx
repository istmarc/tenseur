#ifndef TRENCH_LEASTSQUARES_LINEAR
#define TRENCH_LEASTSQUARES_LINEAR

#include <Ten/Tensor.hxx>
#include <type_traits>

namespace tr{

enum class LSMethod{
   QR = 1,
   LU = 2,
   SVD = 3
};

struct LSOptions{
   const LSMethod method = LSMethod::QR;
};

template<class T = float>
class LS{
   static_assert(std::is_floating_point_v<T>,
      "T must be floating point");
private:
   Matrix<T> _A;
   Vector<T> _b;
   Vector<T> _x;
   LSOptions _options;

public:
   explicit LS(LSOptions options) : _options(options) {}

   /// Solve Ax=b
   void solve(const Matrix<T>& A, const Vector<T>& b) {
   }
};

/// TODO Nonlinear least squares

};

#endif
