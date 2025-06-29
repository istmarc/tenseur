#include "types.hxx"
#include <cmath>
#include <initializer_list>
#include <iostream>

#include <ten/tensor.hxx>
//#include <ten/shared_lib.hxx>

#include <pybind11/pybind11.h>

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

PYBIND11_MODULE(tenseur, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

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
      .def("__repr__", [](const matrix_double& m) {
         std::stringstream ss;
         ss << m;
         return ss.str();
      });

   // Tensor class
   /*py::class_<tensor3_float>(m, "tensor3_float")
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
      .def("__repr__", [](const tensor3_float& t) {
         std::stringstream ss;
         ss << t;
         return ss.str();
      });*/

}

