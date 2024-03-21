# * Find MAGMA library This module finds an installed MAGMA library, a matrix
#   algebra library similar to LAPACK for GPU and multicore systems (see
#   http://icl.cs.utk.edu/magma/).
#
# This module will look for MAGMA library under
# /software/projects/director2178/jorgeg94/mysoft_install by default. To use a
# different installed version of the library set environment variable MAGMA_ROOT
# before running cmake (e.g. MAGMA_ROOT=${HOME}/lib/magma).
#
# This module sets the following variables: MAGMA_FOUND - set to true if the
# MAGMA library is found MAGMA_INCLUDE_DIR - include directory MAGMA_LIBRARIES -
# list of libraries to link against to use MAGMA and creates a library target
# for magma that allows you to simply use it like
# target_link_libraries(your_target magma) to automatically set up the build as
# needed.

if(MAGMA_FOUND)
  return()
endif()

include(FindPackageHandleStandardArgs)

set(MAGMA_INCLUDE_DIR)
find_path(
  MAGMA_INCLUDE_DIR magma.h
  HINTS $ENV{MAGMA_ROOT} /software/projects/director2178/jorgeg94/mysoft_install
  PATH_SUFFIXES include)

set(MAGMA_LIBRARIES)
find_library(
  MAGMA_LIBRARIES magma
  HINTS $ENV{MAGMA_ROOT} /software/projects/director2178/jorgeg94/mysoft_install
  PATH_SUFFIXES lib)

if(MAGMA_INCLUDE_DIR AND MAGMA_LIBRARIES)
  set(MAGMA_FOUND TRUE)
  add_library(magma SHARED IMPORTED)
  set_target_properties(
    magma PROPERTIES IMPORTED_LOCATION "${MAGMA_LIBRARIES}"
                     INTERFACE_INCLUDE_DIRECTORIES "${MAGMA_INCLUDE_DIR}")
  message("-- Found MAGMA: ${MAGMA_LIBRARIES}")
else()
  set(MAGMA_FOUND FALSE)
  message(FATAL_ERROR "-- MAGMA not found, perhaps try setting $MAGMA_ROOT")
endif()
