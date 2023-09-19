# Doxygen
find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)

doxygen_add_docs(
  docs
  "${PROJECT_SOURCE_DIR}/Ten"
  ALL
  COMMENT "Generate docs"
)

# Sphinx
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

set(DOXYGEN_XML_DIR ${CMAKE_CURRENT_BINARY_DIR}/xml)
message("DOXYGEN XML DIR ${}")

add_custom_target(Sphinx
	ALL
	COMMAND
	${SPHINX_EXECUTABLE} -b html
	-Dbreathe_projects.ta=${DOXYGEN_XML_DIR}
	${SPHINX_SOURCE} ${SPHINX_BUILD}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating docs with Sphinx")

