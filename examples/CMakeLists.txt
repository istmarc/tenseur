enable_testing()

# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")

add_subdirectory(Tensor)
add_subdirectory(Expr)
