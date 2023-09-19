cmake_minimum_required(VERSION 3.2)

project(Test VERSION 1.0)

add_executable(test test.cxx)
target_include_directories(test PRIVATE ${PORJECT_SOURCE_DIR})
target_link_directories(test PRIVATE ${PORJECT_SOURCE_DIR})
target_link_libraries(test Tenseur)

