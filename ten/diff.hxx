#ifndef TENSEUR_DIFF
#define TENSEUR_DIFF

#include <functional>

namespace ten{

enum class diff_method{
   default = 1,
   center = 2
};

template<class T = float>
T diff(std::function<T(T)> f, T x, diff_method method = diff_method::center, double h = 1e-8) {
   if (method == diff_method::center) {
      return (f(x + h) - f(x - h) ) / 2*h;
   } else {
      return (f(x) - f(x+h)) / h;
   }
}

}

#endif
