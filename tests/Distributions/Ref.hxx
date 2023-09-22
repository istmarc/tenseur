#ifndef TENSEUR_TESTS_EXPR_REF
#define TENSEUR_TESTS_EXPR_REF

#include <Ten/Types.hxx>

namespace ten::tests {
// Dynamic tensors
template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
auto add(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
auto sub(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] - b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
auto mul(const A &a, const B &b) {
   assert(a.shape() == b.shape() && "Expected equal shapes");
   A c(a.shape());
   for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] * b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value
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
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value &&
            ::ten::isStaticTensor<A>::value &&
            (A::staticSize() == B::staticSize())
auto adds(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::staticSize(); i++) {
      c[i] = a[i] + b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value &&
            ::ten::isStaticTensor<A>::value &&
            (A::staticSize() == B::staticSize())
auto muls(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::staticSize(); i++) {
      c[i] = a[i] * b[i];
   }
   return c;
}

template <class A, class B>
   requires std::same_as<A, B> && ::ten::isDenseVector<A>::value &&
            ::ten::isStaticTensor<A>::value &&
            (A::staticSize() == B::staticSize())
auto divs(const A &a, const B &b) {
   A c;
   for (size_t i = 0; i < A::staticSize(); i++) {
      c[i] = a[i] / b[i];
   }
   return c;
}

} // namespace ten::tests

#endif
