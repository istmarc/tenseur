enable_testing()

# Google tests
include(FetchContent)
FetchContent_Declare(
   googletest
   URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")

# Tests
add_subdirectory(Tensor)
add_subdirectory(Expr)
add_subdirectory(Distributions)
add_subdirectory(Utils)

