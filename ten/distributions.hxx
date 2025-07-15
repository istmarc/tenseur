#ifndef TENSEUR_DISTRIBUTIONS
#define TENSEUR_DISTRIBUTIONS

#include <ten/tensor.hxx>
#include <ten/types.hxx>

#include <random>
#include <type_traits>

namespace ten {

/// Distribution
template <class T> class distribution {
   static_assert(std::is_floating_point_v<T>, "T must be a floating point.");
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
template <class T = float> class uniform : distribution<T> {
 public:
   using value_type = T;
   using dist_type = std::uniform_real_distribution<T>;

 private:
   T _lower_bound;
   T _upper_bound;
   dist_type dist;

 public:
   explicit uniform(T lower_bound = 0., T upper_bound = 1.)
       : _lower_bound(lower_bound), _upper_bound(upper_bound) {
      dist = dist_type(lower_bound, upper_bound);
   }

   T sample() { return dist(::ten::random_engine); }

   vector<T> sample(size_t n) {
      vector<T> s(n);
      for (size_t i = 0; i < n; i++) {
         s[i] = sample();
      }
      return s;
   }
};

/// \class normal
/// Normal distribution
template <class T = float> class normal : distribution<T> {
 public:
   using type = T;
   using dist_type = std::normal_distribution<T>;

 private:
   T _mean;
   T _std;
   dist_type dist;

 public:
   explicit normal(T mean = 0., T std = 1.)
       : _mean(mean), _std(std) {
      dist = dist_type{mean, std};
   }

   T mean() const { return _mean; }

   T std() const { return _std; }

   T sample() { return dist(::ten::random_engine); }

   vector<T> sample(size_t n) {
      vector<T> s(n);
      for (size_t i = 0; i < n; i++) {
         s[i] = sample();
      }
      return s;
   }
};

/// \class mv_normal
/// Multivariate normal distribution
template <class T = float> class mv_normal : distribution<T> {};

/// \class gamma
/// Gamma distribution
template <class T = float> class gamma : distribution<T> {
 public:
   using type = T;
   using dist_type = std::gamma_distribution<T>;

 private:
   T _alpha;
   T _beta;
   dist_type dist;

 public:
   explicit gamma(T alpha, T beta)
       : _alpha(alpha), _beta(beta) {
      dist = dist_type{alpha, beta};
   }

   T alpha() const { return _alpha; }

   T beta() const { return _beta; }

   T sample() { return dist(::ten::random_engine); }

   vector<T> sample(size_t n) {
      vector<T> s(n);
      for (size_t i = 0; i < n; i++) {
         s[i] = sample();
      }
      return s;
   }
};


} // namespace ten

#endif
