#ifndef TENSEUR_LINEAR_ALGEBRA_FACTORIZATION
#define TENSEUR_LINEAR_ALGEBRA_FACTORIZATION

#include <Ten/Types.hxx>

namespace ten{

class QR{
private:
   Matrix<float> _q;
   Matrix<float> _r;

public:
   QR() {};

   void factorize(const Matrix<float>& a) {
      ::ten::kernels::qrFactorization(a, q, r);
   }

   auto q() const {return _q;}

   auto r() const {return _r;}

   auto qr() const {return std::make_tuple(_q, _r);}
};

// TODO LU

// TODO SVD

// TODO Cholesky

}

#endif
