INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING
                    ${CMAKE_CURRENT_SOURCE_DIR})

TRIBITS_ADD_EXECUTABLE_AND_TEST(
    EmbeddedPaperVDP
  SOURCES Tempus_EmbeddedPaper.cpp ${TEUCHOS_STD_UNIT_TEST_MAIN}
  TESTONLYLIBS tempus_test_models
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
    EmbeddedPaperBrusselator
  SOURCES Tempus_EmbeddedPaper_Brusselator.cpp ${TEUCHOS_STD_UNIT_TEST_MAIN}
  TESTONLYLIBS tempus_test_models
  NUM_MPI_PROCS 1
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(Test_EmbeddedPaper_CopyFiles
    DEST_FILES Tempus_EmbeddedPaper_VdP.xml numerical_vdp.bash Tempus_EmbeddedPaper_VdP_numericalTests.xml Tempus_EmbeddedPaper_Brusselator.xml
    EXEDEPS EmbeddedPaperVDP EmbeddedPaperBrusselator
  )
