#ifndef TENSEUR_FUNCTIONAL_TYPES
#define TENSEUR_FUNCTIONAL_TYPES

#include <ten/functional.hxx>
#include <ten/types.hxx>

namespace ten {

////////////////////////////////////////////////////////////////////////////////
// Binary functions

template <class> struct is_mul : std::false_type {};
template <class A, class B, class C>
struct is_mul<::ten::functional::template mul<A, B, C>> : std::true_type {};

template <class> struct is_add : std::false_type {};
template <class A, class B, class C>
struct is_add<::ten::functional::binary_func<
    ::ten::binary_operation::add>::template func<A, B, C>> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// Unary functions

template <class> struct is_sqrt : std::false_type {};
template <class A, class B>
struct is_sqrt<::ten::functional::sqrt<A, B>> : std::true_type {};

template <class> struct is_sqr : std::false_type {};
template <class A, class B>
struct is_sqr<::ten::functional::sqr<A, B>> : std::true_type {};

template <class> struct is_abs : std::false_type {};
template <class A, class B>
struct is_abs<::ten::functional::abs<A, B>> : std::true_type {};

} // namespace ten

#endif
