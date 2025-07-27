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

### Todo
- Pythonizations
- CI/CD with tests
- Sparse tensors
- More special matrices
- Automatic differentiation
- Python documentation
- C++ API documentation
- Inplace operations

### Requirements
- Clang compiler with C++20 support
- CMake
- BLAS library (OpenBlas or BLIS)

## Examples

- Multiply and add

```c++
#include <ten/tensor>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 3});
   auto b = ten::range<ten::matrix<float>>({3, 3});
   auto c = ten::ones<ten::vector<float>>(3);

   ten::vector<float> x = a * b + c;
}
```

- QR factorization

```c++
#include <ten/tensor>
#include <ten/linalg>

int main() {
   auto a = ten::range<matrix<float>>({4, 4});
   ten::linalg::qr qr_fact;
   qr_fact.factorize(a);
   auto [q, r] = qr_fact.factors();

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

