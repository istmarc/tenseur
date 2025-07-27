#include "ten/tensor.hxx"
#include <ten/tensor>

int main(){
   auto x = ten::range<ten::vector<float>>({10}, 1.);
   auto y = ten::range<ten::vector<float>>({10}, 1.);

   {
      auto z = ten::cos(x);
      std::cout << z.eval() << std::endl;
   }


   {
      ten::vector<float> z = ten::cos(x);
      std::cout << z << std::endl;
   }

   {
      auto a = ten::cos(x);
      ten::vector<float> b = a;
      b[0] = 1.0f;
      //std::cout << "b = " << b << std::endl;
      //std::cout << "a = " << a.eval() << std::endl;
      //std::cout << b.node().get() << " " << a.eval().node().get() << std::endl;
   }

   {
      auto a = 2.0f * x;
      std::cout << a.eval() << std::endl;
   }

   {
      auto a = 2.0f + x;
      std::cout << a.eval() << std::endl;
   }

   {
      auto a = x + 2.0f;
      std::cout << a.eval() << std::endl;
   }

   {
      auto a = x - 1.0f;
      std::cout << a.eval() << std::endl;
   }

   {
      auto a = 1.0f - x;
      std::cout << a.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::vector<int32_t>>({10});
      auto y = ten::range<ten::vector<int32_t>>({10});
      auto z = x + y;
      std::cout << z.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::vector<uint32_t>>({10});
      auto y = ten::range<ten::vector<uint32_t>>({10});
      auto z = x + y;
      std::cout << z.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::vector<int64_t>>({10});
      auto y = ten::range<ten::vector<int64_t>>({10});
      auto z = x + y;
      std::cout << z.eval() << std::endl;
   }

   {
      auto x = ten::range<ten::vector<size_t>>({10});
      auto y = ten::range<ten::vector<size_t>>({10});
      auto z = x + y;
      std::cout << z.eval() << std::endl;
   }

}
