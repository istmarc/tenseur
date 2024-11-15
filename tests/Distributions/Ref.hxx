#ifndef TENSEUR_TESTS_EXPR_REF
#define TENSEUR_TESTS_EXPR_REF

#include <ten/types.hxx>

namespace ten::tests {
// Dynamic tensors
template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value
auto add(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value
auto sub(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] - b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value
auto mul(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] * b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value
auto div(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] / b[i];
   }
   return c;
}

// Static tensors
template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value &&
            ::ten::is_stensor<A>::value &&
            (A::static_size() == B::static_size())
auto adds(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::static_size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value &&
            ::ten::is_stensor<A>::value &&
            (A::static_size() == B::static_size())
auto muls(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::static_size(); i++) {
      c[i] = a[i] * b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::is_dense_vector<A>::value &&
            ::ten::is_stensor<A>::value &&
            (A::static_size() == B::static_size())
auto divs(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::static_size(); i++) {
      c[i] = a[i] / b[i];
   }
   return c;
}

} // namespace ten::tests

#endif
