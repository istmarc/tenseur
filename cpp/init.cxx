#include <iostream>
#include <ten/tensor>

int main(){
   {auto x = ten::zeros<ten::stensor<float, 2, 3, 4>>();
   auto y = ten::zeros<ten::vector<float>>({10});
   auto z = ten::zeros<float, ten::shape<ten::dynamic, ten::dynamic>>({2, 3});
   auto t = ten::zeros<float, 3>({2, 3, 4});
   auto r = ten::zeros<float, ten::shape<2, 3>>();
   }

   {
      auto x = ten::range<ten::svector<float, 10>>();
      std::cout << x << std::endl;
   }


}
