#ifndef TENSEUR_RANDOM
#define TENSEUR_RANDOM

#include <ten/distributions.hxx>
#include <ten/types.hxx>

#include <initializer_list>
#include <optional>
#include <random>

namespace ten {
/// Random tensor
/// rand<tensor<...>, ...>(shape)
template <class __t, class __distribution = uniform<typename __t::value_type>>
   requires(::ten::is_dtensor<__t>::value)
auto rand(std::initializer_list<size_t> &&dims) {
   __t x(std::move(dims));
   __distribution dist;
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random tensor with seed
/// rand<tensor<...>, ...>(shape)
template <class __t, class __distribution = uniform<typename __t::value_type>>
   requires(::ten::is_dtensor<__t>::value)
auto rand(std::initializer_list<size_t> &&dims, size_t seed) {
   __t x(std::move(dims));
   __distribution dist(seed);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random tensor
/// rand<t, rank>(shape)
template <class __t, size_type __rank, class __shape = dynamic_shape<__rank>,
          storage_order __order = ::ten::default_order,
          class __storage = default_storage<__t, __shape>,
          class __allocator = typename details::allocator_type<__storage>::type>
auto rand(std::initializer_list<size_t> &&dims) {
   return rand<ranked_tensor<__t, __shape, __order, __storage, __allocator>>(
       std::move(dims));
}

/// Random static tensor
/// rand<stensor<...>>
template <class __t, class __distribution = uniform<typename __t::value_type>>
   requires(::ten::is_stensor<__t>::value)
auto rand() {
   __t x;
   __distribution dist;
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

/// Random static tensor with seed
/// rand<stensor<...>>
template <class __t, class __distribution = uniform<typename __t::value_type>>
   requires(::ten::is_stensor<__t>::value)
auto rand(size_t seed) {
   __t x;
   __distribution dist(seed);
   // Fill with random data
   for (size_t i = 0; i < x.size(); i++) {
      x[i] = dist.sample();
   }
   return x;
}

} // namespace ten

#endif
