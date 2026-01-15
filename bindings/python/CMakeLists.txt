# Project
project(tenseurbackend LANGUAGES CXX)

# Compiler options
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")
# Pybind
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)



pybind11_add_module(tenseurbackend ../../ten/tensor.cxx)

target_link_libraries(tenseurbackend PRIVATE pybind11::module ${BLAS_LIBRARIES})

install(TARGETS tenseurbackend DESTINATION lib)

