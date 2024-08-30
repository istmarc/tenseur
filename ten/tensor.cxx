#include <cmath>
#include <iostream>

#include <ten/tensor.hxx>
#include <ten/shared_lib.hxx>

// #include <Ten/Test.hxx>

namespace ten {

struct Struct {
 private:
   int x = 1234;

 public:
   void testFun();
};

extern "C" void Struct::testFun() {
   std::cout << "Struct::testFun() run" << std::endl;
}

} // namespace ten
