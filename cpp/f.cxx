#include <ten/tensor>

int main() {
   using namespace ten;

   auto a = linear<float, 2>(0, 1, {2, 3});
   save(a, "mat.mtx");

   auto b = range<float>({10});
   save(b, "vec.mtx");
}
