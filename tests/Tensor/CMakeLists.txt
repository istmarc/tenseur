add_executable(TestTensor TestTensor.cxx)

target_link_libraries(TestTensor gtest_main)

target_include_directories(TestTensor
   PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/tests
)
gtest_discover_tests(TestTensor)

