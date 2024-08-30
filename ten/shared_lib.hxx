#ifndef TENSEUR_SHARED_LIB
#define TENSEUR_SHARED_LIB

#include <initializer_list>
#include <memory>
#include <ten/distributions.hxx>
#include <ten/random.hxx>
#include <ten/tensor.hxx>

namespace ten {
// Instantiation for shared library

// shape
/*template<> class shape<dynamic>;
template<> class shape<dynamic, dynamic>;
template<> class shape<dynamic, dynamic, dynamic>;
template<> class shape<dynamic, dynamic, dynamic, dynamic>;
template<> class shape<dynamic, dynamic, dynamic, dynamic, dynamic>;
*/

// ranked_tensor<float>
template <>
class ranked_tensor<float, dynamic_shape<1>, storage_order::col_major,
                    dense_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<2>, storage_order::col_major,
                    dense_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<3>, storage_order::col_major,
                    dense_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<4>, storage_order::col_major,
                    dense_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<5>, storage_order::col_major,
                    dense_storage<float, std::allocator<float>>,
                    std::allocator<float>>;

// ranked_tensor<double>
template <>
class ranked_tensor<double, dynamic_shape<1>, storage_order::col_major,
                    dense_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<2>, storage_order::col_major,
                    dense_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<3>, storage_order::col_major,
                    dense_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<4>, storage_order::col_major,
                    dense_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<5>, storage_order::col_major,
                    dense_storage<double, std::allocator<double>>,
                    std::allocator<double>>;

// ranked_tensor<std::complex<float>>
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<1>, storage_order::col_major,
    dense_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<2>, storage_order::col_major,
    dense_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<3>, storage_order::col_major,
    dense_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<4>, storage_order::col_major,
    dense_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<5>, storage_order::col_major,
    dense_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;

// ranked_tensor<std::complex<double>>
template <>
class ranked_tensor<
    std::complex<double>, dynamic_shape<1>, storage_order::col_major,
    dense_storage<std::complex<double>, std::allocator<std::complex<double>>>,
    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<
    std::complex<double>, dynamic_shape<2>, storage_order::col_major,
    dense_storage<std::complex<double>, std::allocator<std::complex<double>>>,
    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<
    std::complex<double>, dynamic_shape<3>, storage_order::col_major,
    dense_storage<std::complex<double>, std::allocator<std::complex<double>>>,
    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<
    std::complex<double>, dynamic_shape<4>, storage_order::col_major,
    dense_storage<std::complex<double>, std::allocator<std::complex<double>>>,
    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<
    std::complex<double>, dynamic_shape<5>, storage_order::col_major,
    dense_storage<std::complex<double>, std::allocator<std::complex<double>>>,
    std::allocator<std::complex<double>>>;

// diagonal<float>
template <>
class ranked_tensor<float, dynamic_shape<1>, storage_order::col_major,
                    diagonal_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<2>, storage_order::col_major,
                    diagonal_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<3>, storage_order::col_major,
                    diagonal_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<4>, storage_order::col_major,
                    diagonal_storage<float, std::allocator<float>>,
                    std::allocator<float>>;
template <>
class ranked_tensor<float, dynamic_shape<5>, storage_order::col_major,
                    diagonal_storage<float, std::allocator<float>>,
                    std::allocator<float>>;

// diagonal<double>
template <>
class ranked_tensor<double, dynamic_shape<1>, storage_order::col_major,
                    diagonal_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<2>, storage_order::col_major,
                    diagonal_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<3>, storage_order::col_major,
                    diagonal_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<4>, storage_order::col_major,
                    diagonal_storage<double, std::allocator<double>>,
                    std::allocator<double>>;
template <>
class ranked_tensor<double, dynamic_shape<5>, storage_order::col_major,
                    diagonal_storage<double, std::allocator<double>>,
                    std::allocator<double>>;

// diagonal<std::complex<float>>
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<1>, storage_order::col_major,
    diagonal_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<2>, storage_order::col_major,
    diagonal_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<3>, storage_order::col_major,
    diagonal_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<4>, storage_order::col_major,
    diagonal_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;
template <>
class ranked_tensor<
    std::complex<float>, dynamic_shape<5>, storage_order::col_major,
    diagonal_storage<std::complex<float>, std::allocator<std::complex<float>>>,
    std::allocator<std::complex<float>>>;

// diagonal<std::complex<double>>
template <>
class ranked_tensor<std::complex<double>, dynamic_shape<1>,
                    storage_order::col_major,
                    diagonal_storage<std::complex<double>,
                                     std::allocator<std::complex<double>>>,
                    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<std::complex<double>, dynamic_shape<2>,
                    storage_order::col_major,
                    diagonal_storage<std::complex<double>,
                                     std::allocator<std::complex<double>>>,
                    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<std::complex<double>, dynamic_shape<3>,
                    storage_order::col_major,
                    diagonal_storage<std::complex<double>,
                                     std::allocator<std::complex<double>>>,
                    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<std::complex<double>, dynamic_shape<4>,
                    storage_order::col_major,
                    diagonal_storage<std::complex<double>,
                                     std::allocator<std::complex<double>>>,
                    std::allocator<std::complex<double>>>;
template <>
class ranked_tensor<std::complex<double>, dynamic_shape<5>,
                    storage_order::col_major,
                    diagonal_storage<std::complex<double>,
                                     std::allocator<std::complex<double>>>,
                    std::allocator<std::complex<double>>>;

// distributions
template <> class distribution<float>;
template <> class distribution<double>;

template <> class uniform<float>;
template <> class uniform<double>;

template <> class normal<float>;
template <> class normal<double>;

// TODO random

// TODO linear algebra

} // namespace ten

#endif
