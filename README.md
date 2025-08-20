[![docs](https://readthedocs.org/projects/tenseur/badge/?version=latest)](https://tenseur.readthedocs.io/en/latest/index.html)

## Tenseur
A header only C++20 tensor library [WIP]

### Features
- Multi dimensional arrays
- Support static, dynamic and mixed shape tensors
- Lazy evaluation of expressions
- BLAS backend for high performance numerical linear algebra
- Chain expressions
- Factory functions: fill, ones, zeros, range, rand
- Compile to a shared library
- Tests for shared library
- Generate automatic python bindings
- Match and fuse operations
- Inplace operations

### Todo
- Pythonizations
- CI/CD with tests
- Sparse tensors
- More special matrices
- Automatic differentiation
- Python documentation
- C++ API documentation

### Requirements
- Clang compiler with C++20 support
- CMake
- BLAS library (OpenBlas or BLIS)

# Examples

- Assignment

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

- Gemm

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 3});
   auto b = ten::range<ten::matrix<float>>({3, 3});
   auto c = ten::range<ten::matrix<float>>({3, 3});

   // Match and fuse to a single call to gemm
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

