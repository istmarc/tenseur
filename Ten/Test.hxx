#ifndef TENSEUR_T_
#define TENSEUR_T_

#include <iostream>

namespace ten{
static void testFun() {
   std::cout << "testFun()" << std::endl;
}

struct Array{
   size_t _size = 0;
   Array(size_t size) : _size(size) {}
   size_t size() const {
      return _size;
   }
};

}

#endif
