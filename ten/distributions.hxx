#ifndef TENSEUR_DISTRIBUTIONS
#define TENSEUR_DISTRIBUTIONS

#include <ten/tensor.hxx>
#include <ten/types.hxx>

#include <random>
#include <type_traits>

namespace ten {

/// Distribution
template <class __t> class distribution {
   static_assert(std::is_floating_point_v<__t>, "__t must be a floating point.");
};

// FIXME isDistribution
/*struct isDistribution : std::false_type {}
template<class D>
struct isDistribution<D> {
   using value_type = std::is_base_of<D, Distribution>::value;
};*/

static std::random_device random_device;
static std::mt19937 random_engine;

static void set_seed(size_t seed) {
   ::ten::random_engine = decltype(::ten::random_engine)(seed);
}

/// \class uniform
/// Uniform distribution
template <class __t = float> class uniform : distribution<__t> {
 public:
   using value_type = __t;
   using dist_type = std::uniform_real_distribution<__t>;

 private:
   __t _lower_bound;
   __t _upper_bound;
   dist_type dist;

 public:
   explicit uniform(__t lower_bound = 0., __t upper_bound = 1.)
       : _lower_bound(lower_bound), _upper_bound(upper_bound) {
      dist = dist_type(lower_bound, upper_bound);
   }

   __t sample() { return dist(::ten::random_engine); }

   vector<__t> sample(size_t n) {
      vector<__t> s(n);
      for (size_t i = 0; i < n; i++) {
         s[i] = sample();
      }
      return s;
   }
};

/// \class normal
/// Normal distribution
template <class __t = float> class normal : distribution<__t> {
 public:
   using type = __t;
   using dist_type = std::normal_distribution<__t>;

 private:
   __t _mean;
   __t _std;
   dist_type dist;

 public:
   explicit normal(__t mean = 0., __t std = 1.)
       : _mean(mean), _std(std), dist(dist_type(mean, std)) {}

   __t mean() const { return _mean; }

   __t std() const { return _std; }

   __t sample() { return dist(::ten::random_engine); }

   vector<__t> sample(size_t n) {
      vector<__t> s(n);
      for (size_t i = 0; i < n; i++) {
         s[i] = sample();
      }
      return s;
   }
};

/// \class mv_normal
/// Multivariate normal distribution
template <class __t = float> class mv_normal : distribution<__t> {};

} // namespace ten

#endif
