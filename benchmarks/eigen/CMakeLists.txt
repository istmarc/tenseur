find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(EigenBench EigenBench.cxx)
target_include_directories(EigenBench PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(EigenBench nanobench Eigen3::Eigen)
