#include <Ten/Tensor>
#include <Ten/ML/Histogram.hxx>

int main(){
   using namespace ten;

   HistogramOptions options{.standartize = true, .nbins = 10};
   Histogram<> hist(options);

   setSeed(1234);
   Normal<> norm;
   size_t n = 1000;
   Vector<float> x = sample(norm, n);
   save(x, "norm.mtx");

   hist.fit(x);
   auto [h, bins] = hist.histogram();
   std::cout << "hist: " << h << std::endl;
   std::cout << "bins: " << bins << std::endl;
}
