#include "Ten/Storage/DenseStorage.hxx"
#include <ios>
#include <iostream>

#include <Ten/Expr.hxx>
#include <Ten/Functional.hxx>
#include <Ten/Shape.hxx>
#include <Ten/Tensor.hxx>
#include <Ten/UntypedTensor.hxx>
#include <memory>
#include <type_traits>

template <class T> void printTensor(const T &val) {
   std::cout << "[";
   for (size_t i = 0; i < val.size(); i++)
      std::cout << val[i] << " ";
   std::cout << "]\n";
}

int main() {
   using namespace ten;
   using namespace std;

   {
      cout << "Empty tensor" << std::endl;
      Tensor<float, 2> x;
      x.resize({2, 3});
      size_t k = 0;
      for (size_t j = 0; j < 3; j++) {
         for (size_t i = 0; i < 2; i++) {
            x(i, j) = k;
            k++;
         }
      }
      printTensor(x);
   }

   {
      cout << "UntypedTensor" << endl;
      UMatrix x({2, 3});
      std::cout << "float: " << std::boolalpha << x.isOfType<float>() << endl;
      std::cout << "double: " << std::boolalpha << x.isOfType<double>() << endl;
   }

   return 0;
}
