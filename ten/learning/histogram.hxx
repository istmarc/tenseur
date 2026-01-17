#ifndef TENSEUR_LEARNING_HISTOGRAM
#define TENSEUR_LEARNING_HISTOGRAM

#include <ten/tensor>

#include <tuple>
#include <type_traits>

namespace ten {
namespace ml {

struct histogram_options {
   bool standartize = false;
   bool cumulative = false;
   size_t nbins = 0;
};

template <typename T = float> class histogram {
   static_assert(std::is_floating_point_v<T>, "T must be a floating point");

 private:
   ::ten::vector<T> _bins;
   ::ten::vector<T> _hist;
   histogram_options _options;

 public:
   explicit histogram(const histogram_options &options) : _options(options) {}
   explicit histogram(histogram_options &&options) : _options(std::move(options)) {}

   // fit a histogram
   void fit(::ten::vector<T> &data) {
      size_t n = data.size();
      size_t nbins = 0;
      if (_options.nbins == 0) {
         nbins = size_t(std::ceil(std::sqrt(n)));
      } else {
         nbins = _options.nbins;
      }
      _bins = ::ten::vector<T>({nbins + 1});
      _hist = ::ten::zeros<::ten::vector<T>>({nbins});
      // Set bins
      T min = ::ten::min(data).eval().value();
      T max = ::ten::max(data).eval().value();
      T width = (max - min) / T(nbins);
      _bins[0] = min;
      _bins[nbins] = max;
      for (size_t i = 1; i < nbins; i++) {
         _bins[i] = _bins[i - 1] + width;
      }
      // Fit histogram
      for (size_t i = 0; i < n; i++) {
         for (size_t j = 0; j < nbins; j++) {
            if (j == nbins - 1) {
               _hist[j] += 1.;
            } else if (_bins[j] <= data[i] && data[i] < _bins[j + 1]) {
               _hist[j] += 1.;
               break;
            }
         }
      }
      if (_options.standartize) {
         for (size_t i = 0; i < nbins; i++) {
            _hist[i] /= n;
         }
      }
   }

   std::tuple<::ten::vector<T>, ::ten::vector<T>> hist() {
      return std::make_tuple(_hist, _bins);
   }
};

} // namespace ml
} // namespace ten

#endif
