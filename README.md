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

### Todo
- Compile to a shared library
- Tests for shared library
- Generate automatic python bindings
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

## Example
```
#include <ten/tensor>

int main() {
   auto a = ten::range<ten::matrix<float>>({3, 3});
   auto b = ten::range<ten::matrix<float>>({3, 3});
   auto c = ten::ones<ten::vector<float>>(3);

   ten::vector<float> x = a * b + c;
}
```

## Build the examples
```
mkdir build-examples
cd build-examples
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_EXAMPLES=ON
cmake --build . --
```

## Build the tests
```
mkdir build-tests
cd build-tests
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_TESTS=ON
cmake --build . --
```

## Build the docs
```
mkdir build-docs
cd build-docs
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTENSEUR_DOCS=ON
cmake --build . --
```

