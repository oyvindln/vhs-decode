# Find FFTW
# ~~~~~~~~
# Once run this will define:
#
# FFTW_FOUND       = system has FFTW lib
# FFTW_LIBRARY     = full path to the FFTW library
# FFTW_INCLUDE_DIR = where to find headers
#
set(FFTW_LIBRARY_NAMES fftw3 libfftw3 fftw3-3 libfftw3-3 fftw3l libfftw3l fftw3l-3 libfftw3l-3 fftw3q libfftw3q fftw3q-3 libfftw3q-3 )

find_library(FFTW_LIBRARY
  NAMES ${FFTW_LIBRARY_NAMES}
  PATHS
    /usr/lib
	/usr/lib/fftw
	/usr/lib/fftw3
    /usr/local/lib
    /usr/local/lib/fftw
	/usr/local/lib/fftw3
    "$ENV{LIB_DIR}/lib"
    "$ENV{LIB}"
)

FIND_PATH(FFTW_INCLUDE_DIR NAMES fftw3.h PATHS
  /usr/include
  /usr/include/fftw
  /usr/include/fftw3
  /usr/local/include
  /usr/local/include/fftw
  /usr/local/include/fftw3
  "$ENV{LIB_DIR}/include"
  "$ENV{INCLUDE}"
  PATH_SUFFIXES fftw3 fftw
)

IF (FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
  SET(FFTW_FOUND TRUE)
ENDIF (FFTW_INCLUDE_DIR AND FFTW_LIBRARY)

IF (FFTW_FOUND)
    MESSAGE(STATUS "Found fftw3: ${FFTW_LIBRARY}")
ELSE (FFTW_FOUND)
  IF (FFTW_FIND_REQUIRED)
    	MESSAGE(FATAL_ERROR "Could not find fftw3 : ${FFTW_INCLUDE_DIR} - ${FFTW_LIBRARY}")
  ENDIF (FFTW_FIND_REQUIRED)
ENDIF (FFTW_FOUND)