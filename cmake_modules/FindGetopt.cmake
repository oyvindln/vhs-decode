# Based on FindALSA.cmake from cmake project
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2022 Kitware, Inc. and Contributors
# All rights reserved.
# Distributed under the OSI-approved BSD 3-Clause License.  See
# https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGetopt
--------

Find Getopt (Getopt)

Find the Getopt librariy (``Getopt``)

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``Getopt::Getopt``, if
Getopt has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Getopt_FOUND``
  True if Getopt_INCLUDE_DIR & Getopt_LIBRARY are found

``Getopt_LIBRARIES``
  List of libraries when using Getopt.

``Getopt_INCLUDE_DIRS``
  Where to find the Getopt headers.

Cache variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Getopt_INCLUDE_DIR``
  the Getopt include directory

``Getopt_LIBRARY``
  the absolute path of the asound library
#]=======================================================================]

find_path(Getopt_INCLUDE_DIR NAMES getopt.h
          DOC "The Getopt include directory"
)

find_library(Getopt_LIBRARY NAMES getopt
          DOC "The Getopt library"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Getopt
                                  REQUIRED_VARS Getopt_LIBRARY Getopt_INCLUDE_DIR)

if(Getopt_FOUND)
  set( Getopt_LIBRARIES ${Getopt_LIBRARY} )
  set( Getopt_INCLUDE_DIRS ${Getopt_INCLUDE_DIR} )
  if(NOT TARGET Getopt::Getopt)
    add_library(Getopt::Getopt UNKNOWN IMPORTED)
    set_target_properties(Getopt::Getopt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Getopt_INCLUDE_DIRS}")
    set_property(TARGET Getopt::Getopt APPEND PROPERTY IMPORTED_LOCATION "${Getopt_LIBRARY}")
  endif()
endif()

mark_as_advanced(Getopt_INCLUDE_DIR Getopt_LIBRARY)