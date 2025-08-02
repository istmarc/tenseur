#ifndef TENSEUR_FUNCTIONAL_TYPES
#define TENSEUR_FUNCTIONAL_TYPES

#include <ten/functional.hxx>

namespace ten {

template <class> struct is_sqrt : std::false_type {};
template <class A, class B>
struct is_sqrt<::ten::functional::sqrt<A, B>> : std::true_type {};

} // namespace ten

#endif
