enable_testing()

# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")

# Expr
add_executable(Expr Expr.cxx)
target_link_libraries(Expr ${BLAS_LIBRARIES})
target_include_directories(Expr PRIVATE ${PROJECT_SOURCE_DIR})

# Tensor
add_executable(Tensor Tensor.cxx)
target_include_directories(Tensor PRIVATE ${PROJECT_SOURCE_DIR})

