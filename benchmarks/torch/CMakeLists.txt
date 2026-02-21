find_package(Torch REQUIRED)

add_executable(TorchBench TorchBench.cxx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(TorchBench nanobench "${TORCH_LIBRARIES}")

set_property(TARGET TorchBench PROPERTY CXX_STANDARD 17)
