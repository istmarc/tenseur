find_package(BLAS REQUIRED)

add_executable(TenseurBench TenseurBench.cxx)
target_include_directories(TenseurBench PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(TenseurBench nanobench ${BLAS_LIBRARIES})
