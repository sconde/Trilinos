// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_DefaultComm.hpp"

#include "Thyra_VectorStdOps.hpp"

#include "Tempus_StepperFactory.hpp"
#include "Tempus_TimeStepControl.hpp"
#include "Tempus_TimeStepControlStrategyBasicVS.hpp"

#include "../TestModels/SinCosModel.hpp"
#include "../TestModels/DahlquistTestModel.hpp"
#include "../TestUtils/Tempus_ConvergenceTestUtils.hpp"

#include <fstream>
#include <vector>

namespace Tempus_Unit_Test {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;
using Teuchos::ParameterList;
using Teuchos::sublist;
using Teuchos::getParametersFromXmlFile;


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyBasicVS, Default_Construction)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyBasicVS<double>());
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getStepType() != "Variable");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getAmplFactor() != 1.75);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getReductFactor() != 0.5);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMinEta() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMaxEta() != 1.0e+16);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyBasicVS, Full_Construction)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyBasicVS<double>(
    1.33, 0.75, 0.01, 0.05));
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getStepType() != "Variable");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getAmplFactor() != 1.33);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getReductFactor() != 0.75);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMinEta() != 0.01);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMaxEta() != 0.05);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyBasicVS, Create_Constructor)
{
  //Construct ParmeterList for testing.
  auto tscs_temp = rcp(new Tempus::TimeStepControlStrategyBasicVS<double>(
    1.33, 0.75, 0.01, 0.05));

  auto pl = rcp_const_cast<Teuchos::ParameterList>(tscs_temp->getValidParameters());

  auto tscs = Tempus::createTimeStepControlStrategyBasicVS<double>(pl);

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getStepType() != "Variable");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getAmplFactor() != 1.33);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getReductFactor() != 0.75);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMinEta() != 0.01);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMaxEta() != 0.05);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyBasicVS, Accessors)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyBasicVS<double>(
    1.33, 0.75, 0.01, 0.05));
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  tscs->setAmplFactor(1.33);
  tscs->setReductFactor(0.75);
  tscs->setMinEta(0.01);
  tscs->setMaxEta(0.05);

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getStepType() != "Variable");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getAmplFactor() != 1.33);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getReductFactor() != 0.75);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMinEta() != 0.01);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getMaxEta() != 0.05);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyBasicVS, setNextTimeStep)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyBasicVS<double>());
  tscs->setAmplFactor(1.1);
  tscs->setReductFactor(0.5);
  tscs->setMinEta(0.01);
  tscs->setMaxEta(0.05);
  tscs->initialize();
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  // Setup the TimeStepControl --------------------------------
  auto tsc = rcp(new Tempus::TimeStepControl<double>());
  tsc->setTimeStepControlStrategy(tscs);
  tsc->setInitTime(0.0);
  tsc->setFinalTime(10.0);
  tsc->setMinTimeStep (0.01);
  tsc->setInitTimeStep(0.1);
  tsc->setMaxTimeStep (1.0);
  tsc->setFinalIndex(100);
  tsc->initialize();
  TEUCHOS_TEST_FOR_EXCEPT(!tsc->isInitialized());
  Tempus::Status status = Tempus::Status::WORKING;

  // Setup the SolutionHistory --------------------------------
  auto model = rcp(new Tempus_Test::DahlquistTestModel<double>());
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =model->getNominalValues();
  auto icSolution = rcp_const_cast<Thyra::VectorBase<double> >(inArgsIC.get_x());
  auto icState = Tempus::createSolutionStateX<double>(icSolution);
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());

  // Test Reducing timestep
  {
    solutionHistory->addState(icState);
    solutionHistory->getCurrentState()->setTimeStep(0.5);
    solutionHistory->getCurrentState()->setTime(0.0);
    solutionHistory->getCurrentState()->setIndex(0);

    // Set up solution history with two time steps.
    for (int i = 0; i < 2; i++) {
      solutionHistory->initWorkingState();

      tsc->setNextTimeStep(solutionHistory, status);

      { // Mock takeStep
        auto currentState = solutionHistory->getCurrentState();
        auto workingState = solutionHistory->getWorkingState();
        auto xN   = workingState->getX();
        Thyra::Vp_S(xN.ptr(), 1.0);
        workingState->setSolutionStatus(Tempus::Status::PASSED);
        workingState->computeNorms(currentState);
      }

      solutionHistory->promoteWorkingState();
      //auto x = solutionHistory->getCurrentState()->getX();
      //std::cout << "  x = " << get_ele(*(x), 0) << std::endl;
    }

    auto currentState = solutionHistory->getCurrentState();
    TEST_FLOATING_EQUALITY( currentState->getTimeStep(), 0.25, 1.0e-14);
    TEST_FLOATING_EQUALITY( currentState->getTime(), 0.75, 1.0e-14);
  }

  // Test increasing timestep
  {
    solutionHistory->clear();
    solutionHistory->addState(icState);
    solutionHistory->getCurrentState()->setTimeStep(0.5);
    solutionHistory->getCurrentState()->setTime(0.0);
    solutionHistory->getCurrentState()->setIndex(0);

    // Set up solution history with two time steps.
    for (int i = 0; i < 2; i++) {
      solutionHistory->initWorkingState();

      tsc->setNextTimeStep(solutionHistory, status);

      { // Mock takeStep
        auto currentState = solutionHistory->getCurrentState();
        auto workingState = solutionHistory->getWorkingState();
        auto xN   = workingState->getX();
        Thyra::Vp_S(xN.ptr(), 0.0);
        workingState->setSolutionStatus(Tempus::Status::PASSED);
        workingState->computeNorms(currentState);
      }

      solutionHistory->promoteWorkingState();
      //auto x = solutionHistory->getCurrentState()->getX();
      //std::cout << "  x = " << get_ele(*(x), 0) << std::endl;
    }

    auto currentState = solutionHistory->getCurrentState();
    TEST_FLOATING_EQUALITY( currentState->getTimeStep(), 0.55, 1.0e-14);
    TEST_FLOATING_EQUALITY( currentState->getTime(), 1.05, 1.0e-14);
  }
}


} // namespace Tempus_Test
