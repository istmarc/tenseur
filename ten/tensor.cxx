#include "types.hxx"
#include <cmath>
#include <initializer_list>
#include <iostream>

#include <ten/tensor>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
using matrix_float = ten::matrix<float>;
using matrix_double = ten::matrix<double>;
using tensor3_float = ten::tensor<float, 3>;
using tensor3_double = ten::tensor<double, 3>;
using tensor4_float = ten::tensor<float, 4>;
using tensor4_double = ten::tensor<double, 4>;
using tensor5_float = ten::tensor<float, 5>;
using tensor5_double = ten::tensor<double, 5>;

PYBIND11_MODULE(tenseur, m) {
    m.doc() = "Tenseur";

   // Shape
   py::class_<vector_shape>(m, "vector_shape")
      .def(py::init<std::initializer_list<size_t>>())
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
      .value("dfloat", ten::data_type::dfloat)
      .value("ddouble", ten::data_type::ddouble)
      .value("dint32", ten::data_type::dint32)
      .value("dint64", ten::data_type::dint64)
      .value("dcomplexfloat", ten::data_type::dcomplexfloat)
      .value("dcomplexdouble", ten::data_type::dcomplexdouble);

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

   // Vector float
   py::class_<vector_float>(m, "vector_float")
      .def(py::init<std::size_t>())
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
      .def("__repr__", [](const vector_float& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });
   // Vector double
   py::class_<vector_double>(m, "vector_double")
      .def(py::init<std::size_t>())
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
      .def("__repr__", [](const vector_double& v) {
         std::stringstream ss;
         ss << v;
         return ss.str();
      });

   // Matrix float
   py::class_<matrix_float>(m, "matrix_float")
      .def(py::init<std::size_t, std::size_t>())
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




}

