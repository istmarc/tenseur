add_executable(TestUtils TestUtils.cxx)
target_link_libraries(TestUtils gtest_main)

target_include_directories(TestUtils
   PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/tests
)
gtest_discover_tests(TestUtils)

