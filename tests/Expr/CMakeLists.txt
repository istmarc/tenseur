add_executable(TestExpr TestExpr.cxx)
target_link_libraries(TestExpr gtest_main)

target_include_directories(TestExpr
   PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/tests
)
gtest_discover_tests(TestExpr)

