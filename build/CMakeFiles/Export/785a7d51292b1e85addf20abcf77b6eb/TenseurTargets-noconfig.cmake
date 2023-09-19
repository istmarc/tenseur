#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Tenseur::Tenseur" for configuration ""
set_property(TARGET Tenseur::Tenseur APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Tenseur::Tenseur PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libTenseur.so"
  IMPORTED_SONAME_NOCONFIG "libTenseur.so"
  )

list(APPEND _cmake_import_check_targets Tenseur::Tenseur )
list(APPEND _cmake_import_check_files_for_Tenseur::Tenseur "${_IMPORT_PREFIX}/lib/libTenseur.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
