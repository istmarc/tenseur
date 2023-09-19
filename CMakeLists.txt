cmake_minimum_required(VERSION 3.2)

cmake_policy(SET CMP0135 NEW)

project(Tenseur VERSION 1.0.0
                DESCRIPTION "A header only C++20 tensor library."
                HOMEPAGE_URL "https://github.com/istmarco/Tenseur")

include(GNUInstallDirs)

# Compiler options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

# Build shared library
option(TENSEUR_SHAREDLIB "Build shared Library." OFF)

if (TENSEUR_SHAREDLIB)
   add_library(Tenseur SHARED ${PROJECT_SOURCE_DIR}/Ten/Tensor.cxx)
   set_target_properties(Tenseur PROPERTIES POSITION_INDEPENDANT_CODE ON)
   target_include_directories(Tenseur PRIVATE ${PROJECT_SOURCE_DIR})
else()
   add_library(Tenseur INTERFACE)
   target_include_directories(
     Tenseur
     INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Ten>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
   )
   # Compiler features
   target_compile_features(Tenseur INTERFACE cxx_std_20)
endif()

# Target paths
install(TARGETS Tenseur
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Version
include(CMakePackageConfigHelpers)
write_basic_package_version_file("TenseurConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)


configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Targets to install
install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Files to install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake)

# Install directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Ten DESTINATION include)

# Build the tests
option(TENSEUR_TESTS "Build tests." OFF)
if (TENSEUR_TESTS)
  add_subdirectory(tests)
endif()

# Build the examples
option(TENSEUR_EXAMPLES "Build examples." OFF)
if (TENSEUR_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build the docs
option(TENSEUR_DOCS "Build documentation." OFF)
if (TENSEUR_DOCS)
  add_subdirectory(docs)
endif()

# Build the benchmarks
option(TENSEUR_BENCHMARKS "Build documentation." OFF)
if (TENSEUR_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Python bindings
option(TENSEUR_PYTHON "Python bindings." OFF)
if (TENSEUR_PYTHON)
  add_subdirectory(bindings)
endif()

