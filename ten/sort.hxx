#ifndef TENSEUR_SORT
#define TENSEUR_SORT

#include <ten/types.hxx>
#include <ten/tensor.hxx>

#include <type_traits>

namespace ten{
// FIXME Better option?
namespace {
   template<class __t>
   requires(std::is_floating_point_v<T>)
   void quick_sort_inplace(__t* x) {
   }
   // TODO QuickSort
}

enum class sort_method{
   quick_sort = 1,
   merge_sort = 2
};

/// Sort inplace
template<class __t>
requires(::ten::is_dtensor<__t>::value || 
   ::ten::is_stensor<__t>::value)
void sort_inplace(__t& t, sort_method method = sort_method::quick_sort) {
}

/// Sort a tensor
template<class __t>
requires(::ten::is_dtensor<__t>::value)
T sort(const __t& t, sort_method method = sort_method::quick_sort) {
   __t x(t.shape());
   // TODO sort
   return x;
};

/// Sort a static tensor
template<class __t>
requires(::ten::is_stensor<__t>::value)
T sort(const __t& t, sort_method method = sort_method::quick_sort) {
   __t x;
   // TODO sort
   return x;
};

}

#endif
