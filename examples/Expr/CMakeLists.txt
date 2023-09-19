add_executable(Expr Expr.cxx)
target_link_libraries(Expr ${BLAS_LIBRARIES})
target_include_directories(Expr PRIVATE ${PROJECT_SOURCE_DIR})
