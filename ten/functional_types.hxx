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
template <class A, class B, class C>
struct is_mul<::ten::functional::binary_func<
    ten::binary_operation::mul>::template func<A, B, C>> : std::true_type {};
template <class A, class B, class C>
struct is_mul<::ten::functional::scalar_left_binary_func<
    ten::binary_operation::mul>::template func<A, B, C>> : std::true_type {};
template <class A, class B, class C>
struct is_mul<::ten::functional::scalar_right_binary_func<
    ten::binary_operation::mul>::template func<A, B, C>> : std::true_type {};

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

template <class> struct is_pow : std::false_type {};
template <class A, class B>
struct is_pow<::ten::functional::pow<A, B>> : std::true_type {};

template <class> struct is_sum : std::false_type {};
template <class A, class B>
struct is_sum<::ten::functional::sum<A, B>> : std::true_type {};

template <class> struct is_prod : std::false_type {};
template <class A, class B>
struct is_prod<::ten::functional::prod<A, B>> : std::true_type {};

template <class> struct is_sin : std::false_type {};
template <class A, class B>
struct is_sin<::ten::functional::sin<A, B>> : std::true_type {};

template <class> struct is_asin : std::false_type {};
template <class A, class B>
struct is_asin<::ten::functional::asin<A, B>> : std::true_type {};

template <class> struct is_sinh : std::false_type {};
template <class A, class B>
struct is_sinh<::ten::functional::sinh<A, B>> : std::true_type {};

template <class> struct is_cos : std::false_type {};
template <class A, class B>
struct is_cos<::ten::functional::cos<A, B>> : std::true_type {};

template <class> struct is_acos : std::false_type {};
template <class A, class B>
struct is_acos<::ten::functional::acos<A, B>> : std::true_type {};

template <class> struct is_cosh : std::false_type {};
template <class A, class B>
struct is_cosh<::ten::functional::cosh<A, B>> : std::true_type {};

template <class> struct is_tan : std::false_type {};
template <class A, class B>
struct is_tan<::ten::functional::tan<A, B>> : std::true_type {};

template <class> struct is_atan : std::false_type {};
template <class A, class B>
struct is_atan<::ten::functional::atan<A, B>> : std::true_type {};

template <class> struct is_tanh : std::false_type {};
template <class A, class B>
struct is_tanh<::ten::functional::tanh<A, B>> : std::true_type {};

template <class> struct is_exp : std::false_type {};
template <class A, class B>
struct is_exp<::ten::functional::exp<A, B>> : std::true_type {};

template <class> struct is_log : std::false_type {};
template <class A, class B>
struct is_log<::ten::functional::log<A, B>> : std::true_type {};

template <class> struct is_log10 : std::false_type {};
template <class A, class B>
struct is_log10<::ten::functional::log10<A, B>> : std::true_type {};

} // namespace ten

#endif
