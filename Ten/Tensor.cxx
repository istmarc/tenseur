#include <cmath>
#include <iostream>

//#include <Ten/Tensor>
//#include <Ten/SharedLib.hxx>
//#include <Ten/Test.hxx>

namespace ten{

struct Struct{
private:
   int x = 1234;
public:
   void testFun();
};

extern "C" void Struct::testFun() {
   std::cout << "Struct::testFun() run" << std::endl;
}

}
