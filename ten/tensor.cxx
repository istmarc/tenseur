#include <ten/functional.hxx>
#include <ten/types.hxx>

#include <cmath>
#include <initializer_list>
#include <iostream>

#include <ten/tensor>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;


// shape
using vector_shape = ten::shape<0>;
using matrix_shape = ten::shape<0, 0>;
using tensor3_shape = ten::shape<0, 0, 0>;
using tensor4_shape = ten::shape<0, 0, 0, 0>;
using tensor5_shape = ten::shape<0, 0, 0, 0, 0>;

// strides
using vector_stride = ten::stride<ten::shape<0>, ten::storage_order::col_major>;
using matrix_stride = ten::stride<ten::shape<0, 0>, ten::storage_order::col_major>;
using tensor3_stride = ten::stride<ten::shape<0, 0, 0>, ten::storage_order::col_major>;
using tensor4_stride = ten::stride<ten::shape<0, 0, 0, 0>, ten::storage_order::col_major>;
using tensor5_stride = ten::stride<ten::shape<0, 0, 0, 0, 0>, ten::storage_order::col_major>;

// vector, matrix and tensor
using vector_float = ten::vector<float>;
using vector_double = ten::vector<double>;
using vector_int32 = ten::vector<int>;
using vector_int64 = ten::vector<long int>;
using matrix_float = ten::matrix<float>;
using matrix_double = ten::matrix<double>;
using tensor3_float = ten::tensor<float, 3>;
using tensor3_double = ten::tensor<double, 3>;
using tensor4_float = ten::tensor<float, 4>;
using tensor4_double = ten::tensor<double, 4>;
using tensor5_float = ten::tensor<float, 5>;
using tensor5_double = ten::tensor<double, 5>;
using diagonal_float = ten::diagonal<float>;
using diagonal_double = ten::diagonal<double>;

// Scalars
using scalar_float = ten::scalar<float>;
using scalar_double = ten::scalar<double>;
using scalar_int32 = ten::scalar<int>;
using scalar_int64 = ten::scalar<long int>;

// Scalar nodes
using scalarnode_float = scalar_float::node_type;
using scalarnode_double = scalar_double::node_type;
using scalarnode_int32 = scalar_int32::node_type;
using scalarnode_int64 = scalar_int64::node_type;

// Tensor nodes
using vectornode_float = vector_float::node_type;
using vectornode_double = vector_double::node_type;
using vectornode_int32 = vector_int32::node_type;
using vectornode_int64 = vector_int64::node_type;
using matrixnode_float = matrix_float::node_type;
using matrixnode_double = matrix_double::node_type;
using tensor3node_float = tensor3_float::node_type;
using tensor3node_double = tensor3_double::node_type;
using tensor4node_float = tensor4_float::node_type;
using tensor4node_double = tensor4_double::node_type;
using tensor5node_float = tensor5_float::node_type;
using tensor5node_double = tensor5_double::node_type;
using diagonalnode_float = diagonal_float::node_type;
using diagonalnode_double = diagonal_double::node_type;

// TODO Functions

// Unary expr
//using mul_scalar_vector_float = ten::binary_expr<scalarnode_float, vectornode_float, ten::functional::binary_func<ten::binary_operation::mul>::template func>;


// Binary expr
using add_vector_float = ten::binary_expr<vectornode_float, vectornode_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_vector_float = ten::binary_expr<vectornode_float, vectornode_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_vector_float = ten::binary_expr<vectornode_float, vectornode_float,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_vector_float = ten::binary_expr<vectornode_float, vectornode_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_vector_double = ten::binary_expr<vectornode_double, vectornode_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_vector_double = ten::binary_expr<vectornode_double, vectornode_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_vector_double = ten::binary_expr<vectornode_double, vectornode_double,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_vector_double = ten::binary_expr<vectornode_double, vectornode_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_vector_int32 = ten::binary_expr<vectornode_int32, vectornode_int32,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_vector_int32 = ten::binary_expr<vectornode_int32, vectornode_int32,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_vector_int32 = ten::binary_expr<vectornode_int32, vectornode_int32,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_vector_int32 = ten::binary_expr<vectornode_int32, vectornode_int32,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_vector_int64 = ten::binary_expr<vectornode_int64, vectornode_int64,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_vector_int64 = ten::binary_expr<vectornode_int64, vectornode_int64,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_vector_int64 = ten::binary_expr<vectornode_int64, vectornode_int64,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_vector_int64 = ten::binary_expr<vectornode_int64, vectornode_int64,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_matrix_float = ten::binary_expr<matrixnode_float, matrixnode_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_matrix_float = ten::binary_expr<matrixnode_float, matrixnode_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_matrix_float = ten::binary_expr<matrixnode_float, matrixnode_float,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_matrix_float = ten::binary_expr<matrixnode_float, matrixnode_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_matrix_double = ten::binary_expr<matrixnode_double, matrixnode_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_matrix_double = ten::binary_expr<matrixnode_double, matrixnode_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_matrix_double = ten::binary_expr<matrixnode_double, matrixnode_double,
   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_matrix_double = ten::binary_expr<matrixnode_double, matrixnode_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor3_float = ten::binary_expr<tensor3node_float, tensor3node_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor3_float = ten::binary_expr<tensor3node_float, tensor3node_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor3_float = ten::binary_expr<tensor3node_float, tensor3node_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor3_double = ten::binary_expr<tensor3node_double, tensor3node_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor3_double = ten::binary_expr<tensor3node_double, tensor3node_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor3_double = ten::binary_expr<tensor3node_double, tensor3node_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor4_float = ten::binary_expr<tensor4node_float, tensor4node_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor4_float = ten::binary_expr<tensor4node_float, tensor4node_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor4_float = ten::binary_expr<tensor4node_float, tensor4node_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor4_double = ten::binary_expr<tensor4node_double, tensor4node_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor4_double = ten::binary_expr<tensor4node_double, tensor4node_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor4_double = ten::binary_expr<tensor4node_double, tensor4node_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor5_float = ten::binary_expr<tensor5node_float, tensor5node_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor5_float = ten::binary_expr<tensor5node_float, tensor5node_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor5_float = ten::binary_expr<tensor5node_float, tensor5node_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_tensor5_double = ten::binary_expr<tensor5node_double, tensor5node_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_tensor5_double = ten::binary_expr<tensor5node_double, tensor5node_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using div_tensor5_double = ten::binary_expr<tensor5node_double, tensor5node_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_diagonal_float = ten::binary_expr<diagonalnode_float, diagonalnode_float,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_diagonal_float = ten::binary_expr<diagonalnode_float, diagonalnode_float,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
//using mul_diagonal_float = ten::binary_expr<diagonalnode_float, diagonalnode_float,
//   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_diagonal_float = ten::binary_expr<diagonalnode_float, diagonalnode_float,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_diagonal_double = ten::binary_expr<diagonalnode_double, diagonalnode_double,
   ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_diagonal_double = ten::binary_expr<diagonalnode_double, diagonalnode_double,
   ten::functional::binary_func<ten::binary_operation::sub>::template func>;
//using mul_diagonal_double = ten::binary_expr<diagonalnode_double, diagonalnode_double,
//   ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_diagonal_double = ten::binary_expr<diagonalnode_double, diagonalnode_double,
   ten::functional::binary_func<ten::binary_operation::div>::template func>;

// Binary func mul
using mul_vector_float_vector_float = ten::binary_expr<vectornode_float, vectornode_float,
   ten::functional::mul<vectornode_float, vectornode_float, vectornode_float>::template func>;
using mul_vector_double_vector_double = ten::binary_expr<vectornode_double, vectornode_double,
   ten::functional::mul<vectornode_double, vectornode_double, vectornode_double>::template func>;

using mul_vector_int32_vector_int32 = ten::binary_expr<vectornode_int32, vectornode_int32,
   ten::functional::mul<vectornode_int32, vectornode_int32, vectornode_int32>::template func>;
using mul_vector_int64_vector_int64 = ten::binary_expr<vectornode_int64, vectornode_int64,
   ten::functional::mul<vectornode_int64, vectornode_int64, vectornode_int64>::template func>;

using mul_matrix_float_matrix_float = ten::binary_expr<matrixnode_float, matrixnode_float,
   ten::functional::mul<matrixnode_float, matrixnode_float, matrixnode_float>::template func>;
using mul_matrix_double_matrix_double = ten::binary_expr<matrixnode_double, matrixnode_double,
   ten::functional::mul<matrixnode_double, matrixnode_double, matrixnode_double>::template func>;

using mul_matrix_float_vector_float = ten::binary_expr<matrixnode_float, vectornode_float,
   ten::functional::mul<matrixnode_float, vectornode_float, vectornode_float>::template func>;
using mul_matrix_double_vector_double = ten::binary_expr<matrixnode_double, vectornode_double,
   ten::functional::mul<matrixnode_double, vectornode_double, vectornode_double>::template func>;

using mul_scalar_float_vector_float = ten::binary_expr<scalarnode_float, vectornode_float,
   ten::functional::mul<scalarnode_float, vectornode_float, vectornode_float>::template func>;
using mul_scalar_double_vector_double = ten::binary_expr<scalarnode_double, vectornode_double,
   ten::functional::mul<scalarnode_double, vectornode_double, vectornode_double>::template func>;

using mul_scalar_float_matrix_float = ten::binary_expr<scalarnode_float, matrixnode_float,
   ten::functional::mul<scalarnode_float, matrixnode_float, matrixnode_float>::template func>;
using mul_scalar_double_matrix_double = ten::binary_expr<scalarnode_double, matrixnode_double,
   ten::functional::mul<scalarnode_double, matrixnode_double, matrixnode_double>::template func>;

using mul_scalar_float_tensor3_float = ten::binary_expr<scalarnode_float, tensor3node_float,
   ten::functional::mul<scalarnode_float, tensor3node_float, tensor3node_float>::template func>;
using mul_scalar_double_tensor3_double = ten::binary_expr<scalarnode_double, tensor3node_double,
   ten::functional::mul<scalarnode_double, tensor3node_double, tensor3node_double>::template func>;

using mul_scalar_float_tensor4_float = ten::binary_expr<scalarnode_float, tensor4node_float,
   ten::functional::mul<scalarnode_float, tensor4node_float, tensor4node_float>::template func>;
using mul_scalar_double_tensor4_double = ten::binary_expr<scalarnode_double, tensor4node_double,
   ten::functional::mul<scalarnode_double, tensor4node_double, tensor4node_double>::template func>;

using mul_scalar_float_tensor5_float = ten::binary_expr<scalarnode_float, tensor5node_float,
   ten::functional::mul<scalarnode_float, tensor5node_float, tensor5node_float>::template func>;
using mul_scalar_double_tensor5_double = ten::binary_expr<scalarnode_double, tensor5node_double,
   ten::functional::mul<scalarnode_double, tensor5node_double, tensor5node_double>::template func>;

using mul_diagonal_float_diagonal_float = ten::binary_expr<diagonalnode_float, diagonalnode_float,
   ten::functional::mul<diagonalnode_float, diagonalnode_float, diagonalnode_float>::template func>;
using mul_diagonal_double_diagonal_double = ten::binary_expr<diagonalnode_double, diagonalnode_double,
   ten::functional::mul<diagonalnode_double, diagonalnode_double, diagonalnode_double>::template func>;

// Others binary functions

// Initialization
// fill<tensor<...>>(__shape, value)
template <class __t>
   requires(::ten::is_dynamic_tensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto fill_py(typename __t::shape_type shape,
                        typename __t::value_type value) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   __t x(std::forward<shape_type>(shape));
   for (ten::size_type i = 0; i < x.size(); i++) {
      x[i] = value;
   }
   return x;
}

// zeros<tensor<...>>(shape)
template <class __t>
   requires(::ten::is_dynamic_tensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto zeros_py(typename __t::shape_type shape) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   return fill<__t>(std::forward<shape_type>(shape), value_type(0));
}

// ones<tensor<...>>(shape)
template <class __t>
   requires(::ten::is_dynamic_tensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto ones_py(typename __t::shape_type shape) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   return fill<__t>(std::forward<shape_type>(shape), value_type(1));
}

// range<tensor<...>>(__shape, value)
template <class __t>
   requires(::ten::is_dynamic_tensor<__t>::value &&
            ::ten::is_dense_storage<typename __t::storage_type>::value)
[[nodiscard]] auto
range_py(typename __t::shape_type shape,
      typename __t::value_type value = typename __t::value_type(0)) {
   using value_type = typename __t::value_type;
   using shape_type = typename __t::shape_type;
   __t x(std::forward<shape_type>(shape));
   x[0] = value;
   for (ten::size_type i = 1; i < x.size(); i++) {
      x[i] = x[i - 1] + value_type(1);
   }
   return x;
}


PYBIND11_MODULE(tenseurbackend, m) {
    m.doc() = "Tenseur Backend";

   /*py::class_<std::initializer_list<std::size_t>>(m, "list")
      .def(py::init<std::size_t>())
      .def(py::init<std::size_t, std::size_t>())
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>());
   */

   // Scalars
   py::class_<scalar_float>(m, "scalar_float")
      .def(py::init<float>())
      .def("value", &scalar_float::value)
      .def("__repr__", [](const scalar_float& s){
         std::stringstream ss;
         ss << s;
         return ss.str();
      });

   py::class_<scalar_double>(m, "scalar_double")
      .def(py::init<double>())
      .def("value", &scalar_double::value)
      .def("__repr__", [](const scalar_double& s){
         std::stringstream ss;
         ss << s;
         return ss.str();
      });

   py::class_<scalar_int32>(m, "scalar_int32")
      .def(py::init<int>())
      .def("value", &scalar_int32::value)
      .def("__repr__", [](const scalar_int32& s){
         std::stringstream ss;
         ss << s;
         return ss.str();
      });

   py::class_<scalar_int64>(m, "scalar_int64")
      .def(py::init<long int>())
      .def("value", &scalar_int64::value)
      .def("__repr__", [](const scalar_int64& s){
         std::stringstream ss;
         ss << s;
         return ss.str();
      });

   // Shape
   py::class_<vector_shape>(m, "vector_shape")
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init<size_t>())
      .def("rank", &vector_shape::rank)
      .def("size", &vector_shape::size)
      .def("dim", &vector_shape::dim)
      .def("__repr__", [](const vector_shape& s){
         std::stringstream ss;
         ss << s.dim(0);
         return ss.str();
      });

   py::class_<matrix_shape>(m, "matrix_shape")
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init<size_t, size_t>())
      .def("rank", &matrix_shape::rank)
      .def("size", &matrix_shape::size)
      .def("dim", &matrix_shape::dim)
      .def("__repr__", [](const matrix_shape& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1);
         return ss.str();
      });


   py::class_<tensor3_shape>(m, "tensor3_shape")
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init<size_t, size_t, size_t>())
      .def("rank", &tensor3_shape::rank)
      .def("size", &tensor3_shape::size)
      .def("dim", &tensor3_shape::dim)
      .def("__repr__", [](const tensor3_shape& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2);
         return ss.str();
      });

   py::class_<tensor4_shape>(m, "tensor4_shape")
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init<size_t, size_t, size_t, size_t>())
      .def("rank", &tensor4_shape::rank)
      .def("size", &tensor4_shape::size)
      .def("dim", &tensor4_shape::dim)
      .def("__repr__", [](const tensor4_shape& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2)
            << "x" << s.dim(3);
         return ss.str();
      });

   py::class_<tensor5_shape>(m, "tensor5_shape")
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init<size_t, size_t, size_t, size_t, size_t>())
      .def("rank", &tensor5_shape::rank)
      .def("size", &tensor5_shape::size)
      .def("dim", &tensor5_shape::dim)
      .def("__repr__", [](const tensor5_shape& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2)
            << "x" << s.dim(3) << "x" << s.dim(4);
         return ss.str();
      });

   // Strides
   py::class_<vector_stride>(m, "vector_stride")
      .def(py::init<ten::shape<0>>())
      .def("rank", &vector_stride::rank)
      .def("dim", &vector_stride::dim)
      .def("__repr__", [](const vector_stride& s){
         std::stringstream ss;
         ss << s.dim(0);
         return ss.str();
      });

   py::class_<matrix_stride>(m, "matrix_stride")
      .def(py::init<ten::shape<0, 0>>())
      .def("rank", &matrix_stride::rank)
      .def("dim", &matrix_stride::dim)
      .def("__repr__", [](const matrix_stride& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1);
         return ss.str();
      });


   py::class_<tensor3_stride>(m, "tensor3_stride")
      .def(py::init<ten::shape<0, 0, 0>>())
      .def("rank", &tensor3_stride::rank)
      .def("dim", &tensor3_stride::dim)
      .def("__repr__", [](const tensor3_stride& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2);
         return ss.str();
      });

   py::class_<tensor4_stride>(m, "tensor4_stride")
      .def(py::init<ten::shape<0, 0, 0, 0>>())
      .def("rank", &tensor4_stride::rank)
      .def("dim", &tensor4_stride::dim)
      .def("__repr__", [](const tensor4_stride& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2)
            << "x" << s.dim(3);
         return ss.str();
      });

   py::class_<tensor5_stride>(m, "tensor5_stride")
      .def(py::init<ten::shape<0, 0, 0, 0, 0>>())
      .def("rank", &tensor5_stride::rank)
      .def("dim", &tensor5_stride::dim)
      .def("__repr__", [](const tensor5_stride& s){
         std::stringstream ss;
         ss << s.dim(0) << "x" << s.dim(1) << "x" << s.dim(2)
            << "x" << s.dim(3) << "x" << s.dim(4);
         return ss.str();
      });


   // Data type
   py::enum_<ten::data_type>(m, "data_type", py::arithmetic())
      .value("float32", ten::data_type::float32)
      .value("float64", ten::data_type::float64)
      .value("int32", ten::data_type::int32)
      .value("int64", ten::data_type::int64)
      .value("complexfloat", ten::data_type::complexfloat)
      .value("complexdouble", ten::data_type::complexdouble);

   // Format
   py::enum_<ten::storage_format>(m, "storage_format", py::arithmetic())
      .value("dense", ten::storage_format::dense)
      .value("coo", ten::storage_format::coo)
      .value("csc", ten::storage_format::csc)
      .value("csr", ten::storage_format::csr)
      .value("diagonal", ten::storage_format::diagonal)
      .value("lower_tr", ten::storage_format::lower_tr)
      .value("upper_tr", ten::storage_format::upper_tr)
      .value("symmetric", ten::storage_format::symmetric)
      .value("transposed", ten::storage_format::transposed)
      .value("hermitian", ten::storage_format::hermitian);


   // Unary expr
   /*py::class_<mul_scalar_vector_float>(m, "mul_scalar_vector_float")
      .def("value", &mul_scalar_vector_float::value)
      .def("eval", ::&mul_scalar_vector_float::eval);
   */

   // Binar expr
   py::class_<add_vector_float>(m, "add_vector_float")
      .def("value", &add_vector_float::value)
      .def("eval", &add_vector_float::eval);
   py::class_<sub_vector_float>(m, "sub_vector_float")
      .def("value", &sub_vector_float::value)
      .def("eval", &sub_vector_float::eval);
   py::class_<mul_vector_float>(m, "mul_vector_float")
      .def("value", &mul_vector_float::value)
      .def("eval", &mul_vector_float::eval);
   py::class_<div_vector_float>(m, "div_vector_float")
      .def("value", &div_vector_float::value)
      .def("eval", &div_vector_float::eval);

   py::class_<add_vector_double>(m, "add_vector_double")
      .def("value", &add_vector_double::value)
      .def("eval", &add_vector_double::eval);
   py::class_<sub_vector_double>(m, "sub_vector_double")
      .def("value", &sub_vector_double::value)
      .def("eval", &sub_vector_double::eval);
   py::class_<mul_vector_double>(m, "mul_vector_double")
      .def("value", &mul_vector_double::value)
      .def("eval", &mul_vector_double::eval);
   py::class_<div_vector_double>(m, "div_vector_double")
      .def("value", &div_vector_double::value)
      .def("eval", &div_vector_double::eval);

   py::class_<add_vector_int32>(m, "add_vector_int32")
      .def("value", &add_vector_int32::value)
      .def("eval", &add_vector_int32::eval);
   py::class_<sub_vector_int32>(m, "sub_vector_int32")
      .def("value", &sub_vector_int32::value)
      .def("eval", &sub_vector_int32::eval);
   py::class_<mul_vector_int32>(m, "mul_vector_int32")
      .def("value", &mul_vector_int32::value)
      .def("eval", &mul_vector_int32::eval);
   py::class_<div_vector_int32>(m, "div_vector_int32")
      .def("value", &div_vector_int32::value)
      .def("eval", &div_vector_int32::eval);

   py::class_<add_vector_int64>(m, "add_vector_int64")
      .def("value", &add_vector_int64::value)
      .def("eval", &add_vector_int64::eval);
   py::class_<sub_vector_int64>(m, "sub_vector_int64")
      .def("value", &sub_vector_int64::value)
      .def("eval", &sub_vector_int64::eval);
   py::class_<mul_vector_int64>(m, "mul_vector_int64")
      .def("value", &mul_vector_int64::value)
      .def("eval", &mul_vector_int64::eval);
   py::class_<div_vector_int64>(m, "div_vector_int64")
      .def("value", &div_vector_int64::value)
      .def("eval", &div_vector_int64::eval);

   py::class_<add_matrix_float>(m, "add_matrix_float")
      .def("value", &add_matrix_float::value)
      .def("eval", &add_matrix_float::eval);
   py::class_<sub_matrix_float>(m, "sub_matrix_float")
      .def("value", &sub_matrix_float::value)
      .def("eval", &sub_matrix_float::eval);
   py::class_<mul_matrix_float>(m, "mul_matrix_float")
      .def("value", &mul_matrix_float::value)
      .def("eval", &mul_matrix_float::eval);
   py::class_<div_matrix_float>(m, "div_matrix_float")
      .def("value", &div_matrix_float::value)
      .def("eval", &div_matrix_float::eval);

   py::class_<add_matrix_double>(m, "add_matrix_double")
      .def("value", &add_matrix_double::value)
      .def("eval", &add_matrix_double::eval);
   py::class_<sub_matrix_double>(m, "sub_matrix_double")
      .def("value", &sub_matrix_double::value)
      .def("eval", &sub_matrix_double::eval);
   py::class_<mul_matrix_double>(m, "mul_matrix_double")
      .def("value", &mul_matrix_double::value)
      .def("eval", &mul_matrix_double::eval);
   py::class_<div_matrix_double>(m, "div_matrix_double")
      .def("value", &div_matrix_double::value)
      .def("eval", &div_matrix_double::eval);

   py::class_<add_tensor3_float>(m, "add_tensor3_float")
      .def("value", &add_tensor3_float::value)
      .def("eval", &add_tensor3_float::eval);
   py::class_<sub_tensor3_float>(m, "sub_tensor3_float")
      .def("value", &sub_tensor3_float::value)
      .def("eval", &sub_tensor3_float::eval);
   py::class_<div_tensor3_float>(m, "div_tensor3_float")
      .def("value", &div_tensor3_float::value)
      .def("eval", &div_tensor3_float::eval);

   py::class_<add_tensor3_double>(m, "add_tensor3_double")
      .def("value", &add_tensor3_double::value)
      .def("eval", &add_tensor3_double::eval);
   py::class_<sub_tensor3_double>(m, "sub_tensor3_double")
      .def("value", &sub_tensor3_double::value)
      .def("eval", &sub_tensor3_double::eval);
   py::class_<div_tensor3_double>(m, "div_tensor3_double")
      .def("value", &div_tensor3_double::value)
      .def("eval", &div_tensor3_double::eval);

   py::class_<add_tensor4_float>(m, "add_tensor4_float")
      .def("value", &add_tensor4_float::value)
      .def("eval", &add_tensor4_float::eval);
   py::class_<sub_tensor4_float>(m, "sub_tensor4_float")
      .def("value", &sub_tensor4_float::value)
      .def("eval", &sub_tensor4_float::eval);
   py::class_<div_tensor4_float>(m, "div_tensor4_float")
      .def("value", &div_tensor4_float::value)
      .def("eval", &div_tensor4_float::eval);

   py::class_<add_tensor4_double>(m, "add_tensor4_double")
      .def("value", &add_tensor4_double::value)
      .def("eval", &add_tensor4_double::eval);
   py::class_<sub_tensor4_double>(m, "sub_tensor4_double")
      .def("value", &sub_tensor4_double::value)
      .def("eval", &sub_tensor4_double::eval);
   py::class_<div_tensor4_double>(m, "div_tensor4_double")
      .def("value", &div_tensor4_double::value)
      .def("eval", &div_tensor4_double::eval);

   py::class_<add_tensor5_float>(m, "add_tensor5_float")
      .def("value", &add_tensor5_float::value)
      .def("eval", &add_tensor5_float::eval);
   py::class_<sub_tensor5_float>(m, "sub_tensor5_float")
      .def("value", &sub_tensor5_float::value)
      .def("eval", &sub_tensor5_float::eval);
   py::class_<div_tensor5_float>(m, "div_tensor5_float")
      .def("value", &div_tensor5_float::value)
      .def("eval", &div_tensor5_float::eval);

   py::class_<add_tensor5_double>(m, "add_tensor5_double")
      .def("value", &add_tensor5_double::value)
      .def("eval", &add_tensor5_double::eval);
   py::class_<sub_tensor5_double>(m, "sub_tensor5_double")
      .def("value", &sub_tensor5_double::value)
      .def("eval", &sub_tensor5_double::eval);
   py::class_<div_tensor5_double>(m, "div_tensor5_double")
      .def("value", &div_tensor5_double::value)
      .def("eval", &div_tensor5_double::eval);

   py::class_<add_diagonal_float>(m, "add_diagonal_float")
      .def("value", &add_diagonal_float::value)
      .def("eval", &add_diagonal_float::eval);
   py::class_<sub_diagonal_float>(m, "sub_diagonal_float")
      .def("value", &sub_diagonal_float::value)
      .def("eval", &sub_diagonal_float::eval);
   //py::class_<mul_diagonal_float>(m, "mul_diagonal_float")
   //   .def("value", &mul_diagonal_float::value)
   //   .def("eval", &mul_diagonal_float::eval);
   py::class_<div_diagonal_float>(m, "div_diagonal_float")
      .def("value", &div_diagonal_float::value)
      .def("eval", &div_diagonal_float::eval);

   py::class_<add_diagonal_double>(m, "add_diagonal_double")
      .def("value", &add_diagonal_double::value)
      .def("eval", &add_diagonal_double::eval);
   py::class_<sub_diagonal_double>(m, "sub_diagonal_double")
      .def("value", &sub_diagonal_double::value)
      .def("eval", &sub_diagonal_double::eval);
   //py::class_<mul_diagonal_double>(m, "mul_diagonal_double")
   //   .def("value", &mul_diagonal_double::value)
   //   .def("eval", &mul_diagonal_double::eval);
   py::class_<div_diagonal_double>(m, "div_diagonal_double")
      .def("value", &div_diagonal_double::value)
      .def("eval", &div_diagonal_double::eval);

   // Binary expr mul
   py::class_<mul_vector_float_vector_float>(m ,"mul_vector_float_vector_float")
      .def("value", &mul_vector_float_vector_float::value)
      .def("eval",  &mul_vector_float_vector_float::eval);
   py::class_<mul_vector_double_vector_double>(m ,"mul_vector_double_vector_double")
      .def("value", &mul_vector_double_vector_double::value)
      .def("eval",  &mul_vector_double_vector_double::eval);

   py::class_<mul_vector_int32_vector_int32>(m ,"mul_vector_int32_vector_int32")
      .def("value", &mul_vector_int32_vector_int32::value)
      .def("eval",  &mul_vector_int32_vector_int32::eval);
   py::class_<mul_vector_int64_vector_int64>(m ,"mul_vector_int64_vector_int64")
      .def("value", &mul_vector_int64_vector_int64::value)
      .def("eval",  &mul_vector_int64_vector_int64::eval);

   py::class_<mul_matrix_float_matrix_float>(m ,"mul_matrix_float_matrix_float")
      .def("value", &mul_matrix_float_matrix_float::value)
      .def("eval",  &mul_matrix_float_matrix_float::eval);
   py::class_<mul_matrix_double_matrix_double>(m ,"mul_matrix_double_matrix_double")
      .def("value", &mul_matrix_double_matrix_double::value)
      .def("eval",  &mul_matrix_double_matrix_double::eval);

   py::class_<mul_matrix_float_vector_float>(m ,"mul_matrix_float_vector_float")
      .def("value", &mul_matrix_float_vector_float::value)
      .def("eval",  &mul_matrix_float_vector_float::eval);
   py::class_<mul_matrix_double_vector_double>(m ,"mul_matrix_double_vector_double")
      .def("value", &mul_matrix_double_vector_double::value)
      .def("eval",  &mul_matrix_double_vector_double::eval);

   py::class_<mul_scalar_float_vector_float>(m ,"mul_scalar_float_vector_float")
      .def("value", &mul_scalar_float_vector_float::value)
      .def("eval",  &mul_scalar_float_vector_float::eval);
   py::class_<mul_scalar_double_vector_double>(m ,"mul_scalar_double_vector_double")
      .def("value", &mul_scalar_double_vector_double::value)
      .def("eval",  &mul_scalar_double_vector_double::eval);

   py::class_<mul_scalar_float_matrix_float>(m ,"mul_scalar_float_matrix_float")
      .def("value", &mul_scalar_float_matrix_float::value)
      .def("eval",  &mul_scalar_float_matrix_float::eval);
   py::class_<mul_scalar_double_matrix_double>(m ,"mul_scalar_double_matrix_double")
      .def("value", &mul_scalar_double_matrix_double::value)
      .def("eval",  &mul_scalar_double_matrix_double::eval);

   py::class_<mul_scalar_float_tensor3_float>(m ,"mul_scalar_float_tensor3_float")
      .def("value", &mul_scalar_float_tensor3_float::value)
      .def("eval",  &mul_scalar_float_tensor3_float::eval);
   py::class_<mul_scalar_double_tensor3_double>(m ,"mul_scalar_double_tensor3_double")
      .def("value", &mul_scalar_double_tensor3_double::value)
      .def("eval",  &mul_scalar_double_tensor3_double::eval);

   py::class_<mul_scalar_float_tensor4_float>(m ,"mul_scalar_float_tensor4_float")
      .def("value", &mul_scalar_float_tensor4_float::value)
      .def("eval",  &mul_scalar_float_tensor4_float::eval);
   py::class_<mul_scalar_double_tensor4_double>(m ,"mul_scalar_double_tensor4_double")
      .def("value", &mul_scalar_double_tensor4_double::value)
      .def("eval",  &mul_scalar_double_tensor4_double::eval);

   py::class_<mul_scalar_float_tensor5_float>(m ,"mul_scalar_float_tensor5_float")
      .def("value", &mul_scalar_float_tensor5_float::value)
      .def("eval",  &mul_scalar_float_tensor5_float::eval);
   py::class_<mul_scalar_double_tensor5_double>(m ,"mul_scalar_double_tensor5_double")
      .def("value", &mul_scalar_double_tensor5_double::value)
      .def("eval",  &mul_scalar_double_tensor5_double::eval);

   /*
   py::class_<mul_diagonal_float_diagonal_float>(m ,"mul_diagonal_float_diagonal_float")
      .def("value", &mul_diagonal_float_diagonal_float::value)
      .def("eval",  &mul_diagonal_float_diagonal_float::eval);
   py::class_<mul_diagonal_double_diagonal_double>(m ,"mul_diagonal_double_diagonal_double")
      .def("value", &mul_diagonal_double_diagonal_double::value)
      .def("eval",  &mul_diagonal_double_diagonal_double::eval);*/

   // Vector float
   py::class_<vector_float>(m, "vector_float")
      .def(py::init<std::size_t>())
      .def("rank", &vector_float::rank)
      .def("size", &vector_float::size)
      .def("shape", &vector_float::shape)
      .def("strides", &vector_float::strides)
      .def("__getitem__", [](const vector_float& v, size_t index){
         return v[index];
      })
      .def("__setitem__", [](vector_float& v, size_t index, float value){
         v[index] = value;
      })
      .def("__call__", [](const vector_float& v, size_t index) {
         return v[index];
      })
      .def("copy", &vector_float::copy)
      .def("format", &vector_float::format)
      .def("data_type", &vector_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      //.def(float() * py::self)
      .def("__repr__", [](const vector_float& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });

   // Vector double
   py::class_<vector_double>(m, "vector_double")
      .def(py::init<std::size_t>())
      .def("rank", &vector_double::rank)
      .def("size", &vector_double::size)
      .def("strides", &vector_double::strides)
      .def("__getitem__", [](const vector_double& v, size_t index){
         return v[index];
      })
      .def("__setitem__", [](vector_double& v, size_t index, float value){
         v[index] = value;
      })
      .def("__call__", [](const vector_double& v, size_t index) {
         return v[index];
      })
      .def("copy", &vector_double::copy)
      .def("format", &vector_double::format)
      .def("data_type", &vector_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      //.def(double() * py::self)
      .def("__repr__", [](const vector_double& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });

   // Vector int32
   py::class_<vector_int32>(m, "vector_int32")
      .def(py::init<std::size_t>())
      .def("rank", &vector_int32::rank)
      .def("size", &vector_int32::size)
      .def("shape", &vector_int32::shape)
      .def("strides", &vector_int32::strides)
      .def("__getitem__", [](const vector_int32& v, size_t index){
         return v[index];
      })
      .def("__setitem__", [](vector_int32& v, size_t index, int value){
         v[index] = value;
      })
      .def("__call__", [](const vector_int32& v, size_t index) {
         return v[index];
      })
      .def("copy", &vector_int32::copy)
      .def("format", &vector_int32::format)
      .def("data_type", &vector_int32::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      //.def(int32() * py::self)
      .def("__repr__", [](const vector_int32& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });

   // Vector int64
   py::class_<vector_int64>(m, "vector_int64")
      .def(py::init<std::size_t>())
      .def("rank", &vector_int64::rank)
      .def("size", &vector_int64::size)
      .def("shape", &vector_int64::shape)
      .def("strides", &vector_int64::strides)
      .def("__getitem__", [](const vector_int64& v, size_t index){
         return v[index];
      })
      .def("__setitem__", [](vector_int64& v, size_t index, int value){
         v[index] = value;
      })
      .def("__call__", [](const vector_int64& v, size_t index) {
         return v[index];
      })
      .def("copy", &vector_int64::copy)
      .def("format", &vector_int64::format)
      .def("data_type", &vector_int64::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      //.def(int64() * py::self)
      .def("__repr__", [](const vector_int64& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });

   // Matrix float
   py::class_<matrix_float>(m, "matrix_float")
      .def(py::init<std::size_t, std::size_t>())
      .def("rank", &matrix_float::rank)
      .def("size", &matrix_float::size)
      .def("shape", &matrix_float::shape)
      .def("strides", &matrix_float::strides)
      .def("__getitem__", [](const matrix_float& m, size_t index){
         return m[index];
      })
      .def("__setitem__", [](matrix_float& m, size_t index, float value){
         m[index] = value;
      })
      .def("__call__", [](const matrix_float& m, size_t rows, size_t cols) {
         return m(rows, cols);
      })
      .def("set", [](matrix_float& m, size_t rows, size_t cols, float value) {
         m(rows, cols) = value;
      })
      .def("copy", &matrix_float::copy)
      .def("format", &matrix_float::format)
      .def("data_type", &matrix_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def("__mul__", [](matrix_float& self, vector_float&other) {
         return self * other;
      }, py::is_operator())
      /*
      .def("__mul__", [](float f, matrix_float& self) {
         return scalar_float(f) * self;
      }, py::is_operator())*/
      //.def(float() * py::self)
      .def("is_transposed", &matrix_float::is_transposed)
      .def("is_symmetric", &matrix_float::is_symmetric)
      .def("is_hermitian", &matrix_float::is_hermitian)
      .def("is_diagonal", &matrix_float::is_diagonal)
      .def("is_lower_tr", &matrix_float::is_lower_tr)
      .def("is_upper_tr", &matrix_float::is_upper_tr)
      .def("is_sparse_coo", &matrix_float::is_sparse_coo)
      .def("is_sparse_csc", &matrix_float::is_sparse_csc)
      .def("is_sparse_csr", &matrix_float::is_sparse_csr)
      .def("is_sparse", &matrix_float::is_sparse)
      .def("__repr__", [](const matrix_float& m) {
         std::stringstream ss;
         ss << m;
         return ss.str();
      });

   // Matrix double
   py::class_<matrix_double>(m, "matrix_double")
      .def(py::init<std::size_t, std::size_t>())
      .def("rank", &matrix_double::rank)
      .def("size", &matrix_double::size)
      .def("shape", &matrix_double::shape)
      .def("strides", &matrix_double::strides)
      .def("__getitem__", [](const matrix_double& m, size_t index){
         return m[index];
      })
      .def("__setitem__", [](matrix_double& m, size_t index, double value){
         m[index] = value;
      })
      .def("__call__", [](const matrix_double& m, size_t rows, size_t cols) {
         return m(rows, cols);
      })
      .def("set", [](matrix_double& m, size_t rows, size_t cols, double value) {
         m(rows, cols) = value;
      })
      .def("copy", &matrix_double::copy)
      .def("format", &matrix_double::format)
      .def("data_type", &matrix_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def("__mul__", [](matrix_double& self, vector_double&other) {
         return self * other;
      }, py::is_operator())
      //.def(py::self * vector_double)
      //.def(double() * py::self)
      .def("is_transposed", &matrix_double::is_transposed)
      .def("is_symmetric", &matrix_double::is_symmetric)
      .def("is_hermitian", &matrix_double::is_hermitian)
      .def("is_diagonal", &matrix_double::is_diagonal)
      .def("is_lower_tr", &matrix_double::is_lower_tr)
      .def("is_upper_tr", &matrix_double::is_upper_tr)
      .def("is_sparse_coo", &matrix_double::is_sparse_coo)
      .def("is_sparse_csc", &matrix_double::is_sparse_csc)
      .def("is_sparse_csr", &matrix_double::is_sparse_csr)
      .def("is_sparse", &matrix_double::is_sparse)
      .def("__repr__", [](const matrix_double& m) {
         std::stringstream ss;
         ss << m;
         return ss.str();
      });

   // Tensor classes

   // Tensor 3d
   py::class_<tensor3_float>(m, "tensor3_float")
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor3_float::rank)
      .def("size", &tensor3_float::size)
      .def("shape", &tensor3_float::shape)
      .def("strides", &tensor3_float::strides)
      .def("__getitem__", [](const tensor3_float& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor3_float& t, size_t index, float value){
         t[index] = value;
      })
      .def("__call__", [](const tensor3_float& t, size_t i, size_t j, size_t k) {
         return t(i, j, k);
      })
      .def("set", [](tensor3_float& t, size_t i, size_t j, size_t k, float value) {
         t(i, j, k) = value;
      })
      .def("copy", &tensor3_float::copy)
      .def("format", &tensor3_float::format)
      .def("data_type", &tensor3_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(float() * py::self)
      .def("is_transposed", &tensor3_float::is_transposed)
      .def("is_symmetric", &tensor3_float::is_symmetric)
      .def("is_hermitian", &tensor3_float::is_hermitian)
      .def("is_diagonal", &tensor3_float::is_diagonal)
      .def("is_lower_tr", &tensor3_float::is_lower_tr)
      .def("is_upper_tr", &tensor3_float::is_upper_tr)
      .def("is_sparse_coo", &tensor3_float::is_sparse_coo)
      .def("is_sparse_csc", &tensor3_float::is_sparse_csc)
      .def("is_sparse_csr", &tensor3_float::is_sparse_csr)
      .def("is_sparse", &tensor3_float::is_sparse);

   py::class_<tensor3_double>(m, "tensor3_double")
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor3_double::rank)
      .def("size", &tensor3_double::size)
      .def("shape", &tensor3_double::shape)
      .def("strides", &tensor3_double::strides)
      .def("__getitem__", [](const tensor3_double& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor3_double& t, size_t index, double value){
         t[index] = value;
      })
      .def("__call__", [](const tensor3_double& t, size_t i, size_t j, size_t k) {
         return t(i, j, k);
      })
      .def("set", [](tensor3_double& t, size_t i, size_t j, size_t k, double value) {
         t(i, j, k) = value;
      })
      .def("copy", &tensor3_double::copy)
      .def("format", &tensor3_double::format)
      .def("data_type", &tensor3_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(double() * py::self)
      .def("is_transposed", &tensor3_double::is_transposed)
      .def("is_symmetric", &tensor3_double::is_symmetric)
      .def("is_hermitian", &tensor3_double::is_hermitian)
      .def("is_diagonal", &tensor3_double::is_diagonal)
      .def("is_lower_tr", &tensor3_double::is_lower_tr)
      .def("is_upper_tr", &tensor3_double::is_upper_tr)
      .def("is_sparse_coo", &tensor3_double::is_sparse_coo)
      .def("is_sparse_csc", &tensor3_double::is_sparse_csc)
      .def("is_sparse_csr", &tensor3_double::is_sparse_csr)
      .def("is_sparse", &tensor3_double::is_sparse);

   // Tensor 4d
   py::class_<tensor4_float>(m, "tensor4_float")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor4_float::rank)
      .def("size", &tensor4_float::size)
      .def("shape", &tensor4_float::shape)
      .def("strides", &tensor4_float::strides)
      .def("__getitem__", [](const tensor4_float& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor4_float& t, size_t index, float value){
         t[index] = value;
      })
      .def("__call__", [](const tensor4_float& t, size_t i, size_t j, size_t k, size_t l) {
         return t(i, j, k, l);
      })
      .def("set", [](tensor4_float& t, size_t i, size_t j, size_t k, size_t l, float value) {
         t(i, j, k, l) = value;
      })
      .def("copy", &tensor4_float::copy)
      .def("format", &tensor4_float::format)
      .def("data_type", &tensor4_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(float() * py::self)
      .def("is_transposed", &tensor4_float::is_transposed)
      .def("is_symmetric", &tensor4_float::is_symmetric)
      .def("is_hermitian", &tensor4_float::is_hermitian)
      .def("is_diagonal", &tensor4_float::is_diagonal)
      .def("is_lower_tr", &tensor4_float::is_lower_tr)
      .def("is_upper_tr", &tensor4_float::is_upper_tr)
      .def("is_sparse_coo", &tensor4_float::is_sparse_coo)
      .def("is_sparse_csc", &tensor4_float::is_sparse_csc)
      .def("is_sparse_csr", &tensor4_float::is_sparse_csr)
      .def("is_sparse", &tensor4_float::is_sparse);


   py::class_<tensor4_double>(m, "tensor4_double")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor4_double::rank)
      .def("size", &tensor4_double::size)
      .def("shape", &tensor4_double::shape)
      .def("strides", &tensor4_double::strides)
      .def("__getitem__", [](const tensor4_double& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor4_double& t, size_t index, double value){
         t[index] = value;
      })
      .def("__call__", [](const tensor4_double& t, size_t i, size_t j, size_t k, size_t l) {
         return t(i, j, k, l);
      })
      .def("set", [](tensor4_double& t, size_t i, size_t j, size_t k, size_t l, double value) {
         t(i, j, k, l) = value;
      })
      .def("copy", &tensor4_double::copy)
      .def("format", &tensor4_double::format)
      .def("data_type", &tensor4_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(double() * py::self)
      .def("is_transposed", &tensor4_double::is_transposed)
      .def("is_symmetric", &tensor4_double::is_symmetric)
      .def("is_hermitian", &tensor4_double::is_hermitian)
      .def("is_diagonal", &tensor4_double::is_diagonal)
      .def("is_lower_tr", &tensor4_double::is_lower_tr)
      .def("is_upper_tr", &tensor4_double::is_upper_tr)
      .def("is_sparse_coo", &tensor4_double::is_sparse_coo)
      .def("is_sparse_csc", &tensor4_double::is_sparse_csc)
      .def("is_sparse_csr", &tensor4_double::is_sparse_csr)
      .def("is_sparse", &tensor4_double::is_sparse);

   // Tensor 5d
   py::class_<tensor5_float>(m, "tensor5_float")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor5_float::rank)
      .def("size", &tensor5_float::size)
      .def("shape", &tensor5_float::shape)
      .def("strides", &tensor5_float::strides)
      .def("__getitem__", [](const tensor5_float& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor5_float& t, size_t index, float value){
         t[index] = value;
      })
      .def("__call__", [](const tensor5_float& t, size_t i, size_t j, size_t k, size_t l, size_t m) {
         return t(i, j, k, l, m);
      })
      .def("set", [](tensor5_float& t, size_t i, size_t j, size_t k, size_t l, size_t m, float value) {
         t(i, j, k, l, m) = value;
      })
      .def("copy", &tensor5_float::copy)
      .def("format", &tensor5_float::format)
      .def("data_type", &tensor5_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(float() * py::self)
      .def("is_transposed", &tensor5_float::is_transposed)
      .def("is_symmetric", &tensor5_float::is_symmetric)
      .def("is_hermitian", &tensor5_float::is_hermitian)
      .def("is_diagonal", &tensor5_float::is_diagonal)
      .def("is_lower_tr", &tensor5_float::is_lower_tr)
      .def("is_upper_tr", &tensor5_float::is_upper_tr)
      .def("is_sparse_coo", &tensor5_float::is_sparse_coo)
      .def("is_sparse_csc", &tensor5_float::is_sparse_csc)
      .def("is_sparse_csr", &tensor5_float::is_sparse_csr)
      .def("is_sparse", &tensor5_float::is_sparse);

   py::class_<tensor5_double>(m, "tensor5_double")
      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>())
      .def("rank", &tensor5_double::rank)
      .def("size", &tensor5_double::size)
      .def("shape", &tensor5_double::shape)
      .def("strides", &tensor5_double::strides)
      .def("__getitem__", [](const tensor5_double& t, size_t index){
         return t[index];
      })
      .def("__setitem__", [](tensor5_double& t, size_t index, double value){
         t[index] = value;
      })
      .def("__call__", [](const tensor5_double& t, size_t i, size_t j, size_t k, size_t l, size_t m) {
         return t(i, j, k, l, m);
      })
      .def("set", [](tensor5_double& t, size_t i, size_t j, size_t k, size_t l, size_t m, double value) {
         t(i, j, k, l, m) = value;
      })
      .def("copy", &tensor5_double::copy)
      .def("format", &tensor5_double::format)
      .def("data_type", &tensor5_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def(double() * py::self)
      .def("is_transposed", &tensor5_double::is_transposed)
      .def("is_symmetric", &tensor5_double::is_symmetric)
      .def("is_hermitian", &tensor5_double::is_hermitian)
      .def("is_diagonal", &tensor5_double::is_diagonal)
      .def("is_lower_tr", &tensor5_double::is_lower_tr)
      .def("is_upper_tr", &tensor5_double::is_upper_tr)
      .def("is_sparse_coo", &tensor5_double::is_sparse_coo)
      .def("is_sparse_csc", &tensor5_double::is_sparse_csc)
      .def("is_sparse_csr", &tensor5_double::is_sparse_csr)
      .def("is_sparse", &tensor5_double::is_sparse);

   // Diagonal
   py::class_<diagonal_float>(m, "diagonal_float")
      .def(py::init<std::size_t, std::size_t>())
      .def("rank", &diagonal_float::rank)
      .def("size", &diagonal_float::size)
      .def("shape", &diagonal_float::shape)
      .def("strides", &diagonal_float::strides)
      .def("__getitem__", [](const diagonal_float& m, size_t index){
         return m[index];
      })
      .def("__setitem__", [](diagonal_float& m, size_t index, float value){
         m[index] = value;
      })
      .def("__call__", [](const diagonal_float& m, size_t rows, size_t cols) {
         return m(rows, cols);
      })
      .def("set", [](diagonal_float& m, size_t rows, size_t cols, float value) {
         m(rows, cols) = value;
      })
      .def("copy", &diagonal_float::copy)
      .def("format", &diagonal_float::format)
      .def("data_type", &diagonal_float::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def("__mul__", [](diagonal_float& self, vector_float&other) {
      //   return self * other;
      //}, py::is_operator())
      /*
      .def("__mul__", [](float f, diagonal_float& self) {
         return scalar_float(f) * self;
      }, py::is_operator())*/
      //.def(float() * py::self)
      .def("is_transposed", &diagonal_float::is_transposed)
      .def("is_symmetric", &diagonal_float::is_symmetric)
      .def("is_hermitian", &diagonal_float::is_hermitian)
      .def("is_diagonal", &diagonal_float::is_diagonal)
      .def("is_lower_tr", &diagonal_float::is_lower_tr)
      .def("is_upper_tr", &diagonal_float::is_upper_tr)
      .def("is_sparse_coo", &diagonal_float::is_sparse_coo)
      .def("is_sparse_csc", &diagonal_float::is_sparse_csc)
      .def("is_sparse_csr", &diagonal_float::is_sparse_csr)
      .def("is_sparse", &diagonal_float::is_sparse)
      .def("__repr__", [](const diagonal_float& m) {
         std::stringstream ss;
         ss << m;
         return ss.str();
      });

   // diagonal double
   py::class_<diagonal_double>(m, "diagonal_double")
      .def(py::init<std::size_t, std::size_t>())
      .def("rank", &diagonal_double::rank)
      .def("size", &diagonal_double::size)
      .def("shape", &diagonal_double::shape)
      .def("strides", &diagonal_double::strides)
      .def("__getitem__", [](const diagonal_double& m, size_t index){
         return m[index];
      })
      .def("__setitem__", [](diagonal_double& m, size_t index, double value){
         m[index] = value;
      })
      .def("__call__", [](const diagonal_double& m, size_t rows, size_t cols) {
         return m(rows, cols);
      })
      .def("set", [](diagonal_double& m, size_t rows, size_t cols, double value) {
         m(rows, cols) = value;
      })
      .def("copy", &diagonal_double::copy)
      .def("format", &diagonal_double::format)
      .def("data_type", &diagonal_double::data_type)
      .def(py::self + py::self)
      .def(py::self - py::self)
      //.def(py::self * py::self)
      .def(py::self / py::self)
      //.def("__mul__", [](diagonal_double& self, diagonal_double&other) {
      //   return self * other;
      //}, py::is_operator())
      .def("is_transposed", &diagonal_double::is_transposed)
      .def("is_symmetric", &diagonal_double::is_symmetric)
      .def("is_hermitian", &diagonal_double::is_hermitian)
      .def("is_diagonal", &diagonal_double::is_diagonal)
      .def("is_lower_tr", &diagonal_double::is_lower_tr)
      .def("is_upper_tr", &diagonal_double::is_upper_tr)
      .def("is_sparse_coo", &diagonal_double::is_sparse_coo)
      .def("is_sparse_csc", &diagonal_double::is_sparse_csc)
      .def("is_sparse_csr", &diagonal_double::is_sparse_csr)
      .def("is_sparse", &diagonal_double::is_sparse)
      .def("__repr__", [](const diagonal_double& m) {
         std::stringstream ss;
         ss << m;
         return ss.str();
      });

   // Transform diagonal to dense
   m.def("dense_float", &ten::dense<diagonal_float>);
   m.def("dense_double", &ten::dense<diagonal_double>);

   // Transposed, symmetric, lower_tr and upper_tr
   m.def("transposed_float", &ten::transposed<matrix_float>);
   m.def("transposed_double", &ten::transposed<matrix_double>);
   m.def("symmetric_float", &ten::symmetric<matrix_float>);
   m.def("symmetric_double", &ten::symmetric<matrix_double>);
   m.def("lower_tr_float", &ten::lower_tr<matrix_float>);
   m.def("lower_tr_double", &ten::lower_tr<matrix_double>);
   m.def("upper_tr_float", &ten::upper_tr<matrix_float>);
   m.def("upper_tr_double", &ten::upper_tr<matrix_double>);

   // FIXME Cast
   //m.def("cast_vector_float", &ten::cast<vector_double, vector_float>);
   //m.def("cast_vector_double", &ten::cast<vector_float, vector_double>);
   //m.def("cast_matrix_float", &ten::cast<matrix_float, matrix_float>);
   //m.def("cast_matrix_double", &ten::cast<matrix_double, matrix_float>);

   // FIXME Reshape

   // FIXME Flatten
   //m.def("flatten_matrix_float", &ten::flatten<matrix_float>);

   // Initializations
   // FIXME Valeur par défaut pour range
   m.def("fill_vector_float", &fill_py<vector_float>);
   m.def("zeros_vector_float", &zeros_py<vector_float>);
   m.def("ones_vector_float", &ones_py<vector_float>);
   m.def("range_vector_float", &range_py<vector_float>);

   // Save to a file
   m.def("save_vector_float", &ten::save<vector_float>);
   m.def("save_vector_double", &ten::save<vector_double>);
   m.def("save_matrix_float", &ten::save<matrix_float>);
   m.def("save_matrix_double", &ten::save<matrix_double>);

}

