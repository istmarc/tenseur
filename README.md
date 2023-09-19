## Tenseur
A header only C++20 tensor library [WIP]

### Features
- Multi dimensional arrays
- Support static, dynamic and mixed shape tensors
- Lazy evaluation of expressions
- BLAS backend for high performance numerical linear algebra
- Chain expressions
- Factory functions: fill, ones, zeros, iota, rand

### Todo
- Shape and strides for static row major tensors
- Make raw major default?
- Compile to a shared library
- Tests for shared library
- Generate automatic python bindings
- Pythonizations
- Tests
- CI/CD
- Check tensor shapes at compile time whenever possible
- Sparse tensors
- Special matrices
- Automatic differentiation
- Python documentation
- C++ API documentation
- Untyped tensor
- Operators precedence
- Inplace operations

### Requirements
- Clang compiler with C++20 support
- CMake
- BLAS library (OpenBlas or BLIS)

## Example
```
#include <Ten/Tensor>

using namespace ten;

int main() {
   auto a = iota<Matrix<float>>({3, 3});
   auto b = iota<Matrix<float>>({3, 3});
   auto c = ones<Vector<float>>(3);

   Vector<float> x = a * b + c;
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

