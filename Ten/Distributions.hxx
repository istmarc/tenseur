#ifndef TENSEUR_DISTRIBUTIONS_DISTRIBUTION_HXX
#define TENSEUR_DISTRIBUTIONS_DISTRIBUTION_HXX

#include "Ten/Tensor.hxx"
#include "Ten/Types.hxx"

#include <random>

namespace ten {

// Distribution can be static of dynamic
class Distribution {};

// sample(distribution)
template <class D> auto sample(D &&distribution) {
   return distribution.nextSample();
}

// sample(distribution, n)
template <class D> Vector<float> sample(D &&distribution, size_t n) {
   Vector<float> s(n);
   for (size_type i = 0; i < n; i++) {
      s[i] = distribution.nextSample();
   }
   return s;
}

// Uniform distribution
template <class T = float> class Uniform : Distribution {
 public:
   using type = T;
   using dist_type = std::uniform_real_distribution<float>;
   using random_engine_type = std::mt19937;

 private:
   T _lowerBound;
   T _upperBound;
   std::random_device randomDevice;
   std::mt19937 randomEngine;
   dist_type dist;

   T nextSample() { return dist(randomEngine); }

 public:
   explicit Uniform(T lowerBound = 0., T upperBound = 1.)
       : _lowerBound(lowerBound), _upperBound(upperBound), randomDevice() {
      randomEngine = random_engine_type(randomDevice());
      dist = dist_type(lowerBound, upperBound);
   }

   template <class D> friend auto sample(D &&);
   template <class D> friend Vector<T> sample(D &&, size_type);
};

// Normal distribution
template <class T = float> class Normal : Distribution {
 public:
   using type = T;
   using dist_type = std::normal_distribution<T>;
   using random_engine_type = std::mt19937;

 private:
   T _mean;
   T _std;
   std::random_device randomDevice;
   std::mt19937 randomEngine;
   dist_type dist;

   T nextSample() { return dist(randomEngine); }

 public:
   explicit Normal(T mean = 0., T std = 1.)
       : _mean(mean), _std(std), randomDevice(),
         randomEngine(random_engine_type(randomDevice())),
         dist(dist_type(mean, std)) {}

   template <class D> friend auto sample(D &&);
   template <class D> friend Vector<T> sample(D &&, size_type);
};

} // namespace ten

#endif
