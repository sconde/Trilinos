
ADD_SUBDIRECTORY(stk_percept)

IF(  TPL_ENABLE_Netcdf)
#  IF (NOT CMAKE_BUILD_TYPE STREQUAL DEBUG)
  IF (NOT ${PARENT_PACKAGE_NAME}_ENABLE_DEBUG)
    TRIBITS_ADD_TEST_DIRECTORIES(unit_tests)
#    TRIBITS_ADD_TEST_DIRECTORIES(regression_tests)
  ENDIF()
ENDIF()
