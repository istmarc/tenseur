enable_testing()

# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")

find_package(LAPACK REQUIRED)
message("LAPACK libraries: ${LAPACK_LIBRARIES}")

# Expr
add_executable(Expr Expr.cxx)
target_link_libraries(Expr ${BLAS_LIBRARIES})
target_include_directories(Expr PRIVATE ${PROJECT_SOURCE_DIR})

# Tensor
add_executable(Tensor Tensor.cxx)
target_include_directories(Tensor PRIVATE ${PROJECT_SOURCE_DIR})

# NeuralNetworksXOR
add_executable(NeuralNetworksXOR NeuralNetworksXOR.cxx)
target_link_libraries(NeuralNetworksXOR ${BLAS_LIBRARIES})
target_include_directories(NeuralNetworksXOR PRIVATE ${PROJECT_SOURCE_DIR})

# SolveLinearSystem
add_executable(SolveLinearSystem SolveLinearSystem.cxx)
target_link_libraries(SolveLinearSystem ${BLAS_LIBRARIES})
target_link_libraries(SolveLinearSystem ${LAPACK_LIBRARIES} -llapacke)
target_include_directories(SolveLinearSystem PRIVATE ${PROJECT_SOURCE_DIR})

# QR
add_executable(QR QR.cxx)
target_link_libraries(QR ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} -llapacke)
target_include_directories(QR PRIVATE ${PROJECT_SOURCE_DIR})

# LU
add_executable(LU LU.cxx)
target_link_libraries(LU ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} -llapacke)
target_include_directories(LU PRIVATE ${PROJECT_SOURCE_DIR})

# Cholesky
add_executable(Cholesky Cholesky.cxx)
target_link_libraries(Cholesky ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} -llapacke)
target_include_directories(Cholesky PRIVATE ${PROJECT_SOURCE_DIR})

# SVD
add_executable(SVD SVD.cxx)
target_link_libraries(SVD ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} -llapacke)
target_include_directories(SVD PRIVATE ${PROJECT_SOURCE_DIR})

