#ifndef TENSEUR_RANDOM
#define TENSEUR_RANDOM

#include <ten/distributions.hxx>
#include <ten/types.hxx>

#include <initializer_list>
#include <optional>
#include <random>

namespace ten {

/// Random normal tensor
/// rand_norm<tensor<...>, ...>(shape)
template <DynamicTensor T>
auto rand_norm(std::initializer_list<size_t> &&dims,
               const typename T::value_type mean = 0.,
               const typename T::value_type std = 1.) {
   T x(std::move(dims));
   using value_type = typename T::value_type;

   ten::normal<value_type> dist(mean, std);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random tensor
/// rand_norm<t, rank>(shape)
template <class T, size_type Rank, class Shape = dynamic_shape<Rank>,
          storage_order Order = ::ten::default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
auto rand_norm(std::initializer_list<size_t> &&dims, const T mean = 0.,
               const T std = 1.) {
   return rand_norm<ranked_tensor<T, Shape, Order, Storage, Allocator>>(
       std::move(dims), mean, std);
}

/// Random static tensor
/// rand_norm<stensor<...>>
template <StaticTensor T>
auto rand_norm(const typename T::value_type mean = 0.,
               const typename T::value_type std = 1.) {
   T x;
   using value_type = typename T::value_type;

   ten::normal<value_type> dist(mean, std);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random uniform tensor
/// rand_unif<tensor<...>, ...>(shape)
template <DynamicTensor T>
auto rand_unif(std::initializer_list<size_t> &&dims,
               const typename T::value_type lower_bound = 0.,
               const typename T::value_type upper_bound = 1.) {
   T x(std::move(dims));
   using value_type = typename T::value_type;

   ten::uniform<value_type> dist(lower_bound, upper_bound);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random tensor
/// rand_norm<t, rank>(shape)
template <class T, size_type Rank, class Shape = dynamic_shape<Rank>,
          storage_order Order = ::ten::default_order,
          class Storage = default_storage<T, Shape>,
          class Allocator = typename details::allocator_type<Storage>::type>
auto rand_unif(std::initializer_list<size_t> &&dims, const T lower_bound = 0.,
               const T upper_bound = 1.) {
   return rand_norm<ranked_tensor<T, Shape, Order, Storage, Allocator>>(
       std::move(dims), lower_bound, upper_bound);
}

/// Random static tensor
/// rand_norm<stensor<...>>
template <StaticTensor T>
auto rand_unif(const typename T::value_type lower_bound = 0.,
               const typename T::value_type upper_bound = 1.) {
   T x;
   using value_type = typename T::value_type;

   ten::uniform<value_type> dist(lower_bound, upper_bound);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

} // namespace ten

#endif
