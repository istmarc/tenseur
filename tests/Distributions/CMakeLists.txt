add_executable(TestDistributions TestDistributions.cxx)
target_link_libraries(TestDistributions gtest_main)

target_include_directories(TestDistributions
   PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/tests
)
gtest_discover_tests(TestDistributions)

