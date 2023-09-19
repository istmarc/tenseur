#ifndef TENSEUR_RANDOM
#define TENSEUR_RANDOM

#include <Ten/Types.hxx>

#include <initializer_list>
#include <optional>
#include <random>

namespace ten::details {
template <class RandomEngine>
auto getEngine = [](const std::optional<size_t> seed) -> decltype(auto) {
   if (seed.has_value()) {
      RandomEngine engine(seed.value());
      return engine;
   } else {
      std::random_device device;
      RandomEngine engine(device());
      return engine;
   }
};
}

namespace ten {
// Random tensor
// rand<Tensor<...>, ...>
template <class T,
          class Distribution = std::normal_distribution<typename T::value_type>,
          class RandomEngine = std::mt19937>
   requires(::ten::isDynamicTensor<T>::value)
auto rand(std::initializer_list<size_t> &&dims,
          const std::optional<size_t> seed = std::nullopt) {
   T x(std::move(dims));
   auto engine = details::getEngine<RandomEngine>(seed);
   Distribution dist;
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist(engine);
   }
   return x;
}

// rand<T, Rank, Shape = default, Storage = defaul, Allocator = default>
template <class T,
   size_type Rank,
   class Shape = DynamicShape<Rank>,
   StorageOrder Order = ::ten::defaultOrder,
   class Storage = ::ten::DefaultStorage<T, Shape>,
   class Allocator = typename details::AllocatorType<Storage>::type>
auto rand(std::initializer_list<size_t> &&dims,
   const std::optional<size_t> seed = std::nullopt) {
   return rand<RankedTensor<T, Shape, Order, Storage, Allocator>>(std::move(dims), seed);
}

// Random static tensor
template <class T,
          class Distribution = std::normal_distribution<typename T::value_type>,
          class RandomEngine = std::mt19937>
   requires(::ten::isStaticTensor<T>::value)
auto rand(const std::optional<size_t> seed = std::nullopt) {
   T x;
   auto engine = details::getEngine<RandomEngine>(seed);
   Distribution dist;
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist(engine);
   }
   return x;
}
} // namespace ten

#endif
