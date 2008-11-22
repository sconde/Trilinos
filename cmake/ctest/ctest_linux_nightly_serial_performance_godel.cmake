
SET(CTEST_SOURCE_NAME Trilinos)
SET(CTEST_BINARY_NAME BUILD)
SET(TEST_TYPE nightly)
SET(BUILD_TYPE release)
SET(EXTRA_BUILD_TYPE serial-performance)
SET(HOSTTYPE Linux) # Have to set this manually on this machine for some reason?
SET(CTEST_DASHBOARD_ROOT /home/rabartl/PROJECTS/dashboards/Trilinos/SERIAL_PERFORMANCE)
SET(CTEST_CMAKE_COMMAND /usr/local/bin/cmake)
SET(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 1000)
SET(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 10000)

# Options for Nightly builds

SET(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)
#SET(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)

SET(CTEST_CVS_CHECKOUT
  "cvs -q -d :ext:software.sandia.gov:/space/CVS co ${CTEST_SOURCE_NAME}"
)
SET (CTEST_CVS_COMMAND
  "cvs -q -d :ext:software.sandia.gov:/space/CVS co ${CTEST_SOURCE_NAME}"
)

SET(CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
SET(CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

SET(CTEST_COMMAND 
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlyStart"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlyUpdate"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlyConfigure"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlyBuild"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlySubmit"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlyTest"
  "\"${CTEST_EXECUTABLE_NAME}\" -D NightlySubmit -A \"${CTEST_BINARY_DIRECTORY}/CMakeCache.txt\;${CTEST_DASHBOARD_ROOT}/../scripts/ctest_linux_nightly_serial_performance_godel.cmake\""
)

SET(CTEST_INITIAL_CACHE "

BUILDNAME:STRING=${HOSTTYPE}-${TEST_TYPE}-${EXTRA_BUILD_TYPE}-${BUILD_TYPE}
CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}

Trilinos_ENABLE_DEPENCENCY_UNIT_TESTS:BOOL=OFF

Trilinos_VERBOSE_CONFIGURE:BOOL=ON

CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++
CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc
CMAKE_Fortran_COMPILER:FILEPATH=/usr/bin/gfortran
HAVE_GCC_ABI_DEMANGLE:BOOL=ON
CMAKE_CXX_FLAGS:STRING=-O3 -ansi -Wall -Wshadow -Wunused-variable -Wunused-function -Wno-system-headers -Wno-deprecated -Woverloaded-virtual -Wwrite-strings -fprofile-arcs -ftest-coverage -fexceptions
CMAKE_C_FLAGS:STRING=-O3 -Wall -fprofile-arcs -ftest-coverage -fexceptions
CMAKE_Fortran_FLAGS:STRING=-O5
CMAKE_EXE_LINKER_FLAGS:STRING=-fprofile-arcs -ftest-coverage -lm
MAKECOMMAND:STRING=gmake -j8 -i

DART_TESTING_TIMEOUT:STRING=600
CMAKE_VERBOSE_MAKEFILE:BOOL=TRUE

CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE:STRING=1000
CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE:STRING=10000

Trilinos_ENABLE_Teuchos:BOOL=ON
Teuchos_ENABLE_COMPLEX:BOOL=ON
Teuchos_ENABLE_EXTENDED:BOOL=ON
Teuchos_ENABLE_BOOST:BOOL=ON
Teuchos_ENABLE_GCC_DEMANGLE:BOOL=ON
Teuchos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON
Teuchos_ENABLE_PERFORMANCE_TESTS:BOOL=ON

EpetraExt_BUILD_GRAPH_REORDERINGS:BOOL=ON
EpetraExt_BUILD_BDF:BOOL=ON

")
