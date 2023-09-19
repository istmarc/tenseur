# Sources

# Headers

# BLAS
# BLAS
find_package(BLAS REQUIRED)
message("BLAS linker flags : ${BLAS_LINKER_FLAGS}")
message("BLAS libraries : ${BLAS_LIBRARIES}")
message("BLAS vendor : ${BLA_VENDOR}")

# Cppyy

execute_process(COMMAND cling-config --cmake OUTPUT_VARIABLE CPPYY_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
# Make sure this is set to something like:
# ~/.local/lib/python2.7/site-packages/cpyy_backend/cmake 
message("CPYY_MODULE_PATH: " ${CPPYY_MODULE_PATH})

list(INSERT CMAKE_MODULE_PATH 0 ${CPPYY_MODULE_PATH})
find_package(Cppyy)

set(EXTRA_CLING_ARGS "--std=c++2b -Wc++2b-extensions -O2 -march=native ")

cppyy_add_bindings(
   "Tenseur" "0.0.0" "Ahmat" ""
   LANGUAGE_STANDARD "20"
   GENERATE_OPTIONS "-D__PIC__;-Wno-macro-redefined"
   INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
   # LINK_LIBRARIES Tenseur
   # LINK_LIBRARIES ${BLAS_LIBRARIES}
   H_DIRS ${PROJECT_SOURCE_DIR}/Ten
   H_FILES "Shape.hxx"
)

