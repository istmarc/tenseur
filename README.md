[![docs](https://readthedocs.org/projects/tenseur/badge/?version=latest)](https://tenseur.readthedocs.io/en/latest/index.html)

# Tenseur

A header only C++23 tensor and mathematical library [WIP]

Tenseur is a header only C++23 tensor library designed for high performance numerical computations, prioritizing speed above all else. It assume that the user will ensure the correctness of their program. Execptions handling and bounds checking are disabled to minimize overhead. This makes it ideal for applications where computational efficiency is the goal such as deep learning and scientific computing. It has also support for automatic differentiation.

## Tensor classes

The library is build around a core tensor class `ranked_tensor<T, class Shape, storge_order order, class Storage, class Allocator>` that provide efficient storage (column major and row major) and manipulation of multidimentional arrays through operator overloading. Operations are implemented using techniques such as SIMD instructions and cache friendly memory access. Error handling is minimized, instead of throwing exceptions, the library relies on the user to validate inputs and manage potential issues. Some check are done at compile time for static tensors (tensors with static shape). Aliases such as `tensor<T, Shape>`, `matrix<T, Shape>`, `vector<T>`, `stensor<T, Dims...>`, `smatrix<T, Rows, Cols>`, and `svector<T, Size>` are defined.

## Expressions API

An expression API class for representing unary and binary operations between tensors is implemented. Its inspired by compiler optimization techniques and passes. It makes it possible to do expression matching at compile time and fuse some opeations, for examples a basic call to gemm can be written as `c = a * b + c`, it will be lowered to `gemm(1.0, a, b, 1.0, c);` instead of writing `c.noalis() = a * b + c` as in most numerical libraries. Also unary operation will be lowered to inplace operations whenever that's possible. For example `c = ten::sqrt(c)` will be lowered to an inplace operation `inplace_sqrt(c)`.

## Features

- [x] Multi dimensional arrays

- [x] Support static, dynamic and mixed shape tensors

- [x] Lazy evaluation of expressions

- [x] BLAS backend for high performance numerical linear algebra

- [x] Automatic differentiation

- [x] Chain expressions

- [x] Factory functions: fill, ones, zeros, range, rand

- [] Compile to a shared library (support for up to 5d dynamic tensors)

- [] Tests for shared library

- [x] Generate automatic python bindings

- [x] Python bindings ([tenseurpy](https://github.com/istmarc/tenseurpy))

- [x] Match and fuse operations

- [x] Inplace operations

- [] Sparse tensors


## Todo

- [] Pythonizations

- [] CI/CD with tests

- [] More special matrices

- [] Python documentation

- [] C++ API documentation

- [] Neural networks

## Requirements

- Clang compiler with C++20 support
- CMake
- BLAS library (OpenBlas or BLIS)

# Examples

- Tensors

```c++
// Dynamic uninitialized tensors
ten::tensor<float, 3> x({2, 3, 4});
// Static tensor with uninitialized memory (no allocations)
ten::stensor<float, 2, 3, 4> y;
// Access indices
x(0, 1, 2) = 3.0f;
std::cout << x(0, 1, 2) << std::endl;
// Slicing using seq
using ten::seq;
auto slice = x(seq(0, last), seq(0, last), seq(1, 2));
// Assign to slice
slice = 1.0f;
// Slicing using mdseq
using ten::mdseq;
auto index = mdseq<3>(seq(0, last), seq(0, last), seq(0, 1));
auto second_slice = x[index];
second_slice = 2.0f;
```

- Assignment, and rows and columns

```c++
ten::matrix<float> x({3, 3});
// Assign a single value
x = 1.0f;
// Row and columns can be accessed using col and row
x.row(0) = 2.0f;
x.col(0) = 3.0f;
std::cout << x << std::endl;
```

- Slicing

```c++
using ten::seq;
using ten::last;
ten::smatrix<float, 3, 3> x;
auto slice = x(seq(1, last), seq(0, 1));
slice = 99.0f;
std::cout << x << std::endl;
```

- Gemm with expressions matching

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 3});
   auto b = ten::range<ten::matrix<float>>({3, 3});
   auto c = ten::range<ten::matrix<float>>({3, 3});

   c = a * b + c;
   std::cout << c << std::endl;
}
```

- QR factorization

```c++
#include <ten/tensor>
#include <ten/linalg>

int main() {
   auto a = ten::range<matrix<float>>({4, 4});
   auto [q, r] = ten::linalg::qr(a);

   std::cout << q << std::endl;
   std::cout << r << std::endl;
}
```

- Automatic differentiation

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   ten::vector<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
   ten::vector<float> y({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
   auto z = x / y;
   auto t = ten::sum(z);
   t.eval();
   t.backward();
   std::cout << "Output = " << t.value() << std::endl;
   std::cout << "The gradients\n";
   std::cout << x.grad() << std::endl;
   std::cout << y.grad() << std::endl;
}
```

- Save and load binary data

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::matrix<float>>({3, 4});
   ten::io::save(x, "matrix.ten");
   auto y = ten::io::load<ten::matrix<float>>("matrix.ten").value();
   std::cout << "shape = " << y.shape() << std::endl;
   std::cout << "stride = " << y.strides() << std::endl;
   std::cout << "data = \n" << y << std::endl;
}
```

- Data frame

```c++
#include <ten/dataframe>

int main() {
    auto df = ten::read_csv("path/to/file.csv");
    std::cout << df << std::endl;
    // Select by column name
    std::cout << df[{"x"}] << std::endl;
    // Select by column index
    std::cout << df[{0, 1, 4}] << std::endl;
    // Select by row index and column name
    std::cout << df.select(ten::seq(0, 2), std::vector<std::string>{"x", "y"}) << std::endl;
}
```

# Building tenseur

## Build the examples

```shell
mkdir build-examples
cd build-examples
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_EXAMPLES=ON
cmake --build . --
```

## Build the tests

```shell
mkdir build-tests
cd build-tests
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_TESTS=ON
cmake --build . --
```

## Build the docs

```shell
mkdir build-docs
cd build-docs
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_DOCS=ON
cmake --build . --
```
## Build and install the python bindings

```shell
mkdir build-bindings
cd build-bindings
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_PYTHON=ON
cmake --build . --
sudo make install
```


