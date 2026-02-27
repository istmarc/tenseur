add_executable(TestLinearAlgebra TestLinearAlgebra.cxx)

target_link_libraries(TestLinearAlgebra gtest_main ${BLAS_LIBRARIES} ${LAPACKE})

target_include_directories(TestLinearAlgebra
   PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/tests
)
gtest_discover_tests(TestLinearAlgebra)

