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
#include "Tempus_TimeStepControlStrategyIntegralController.hpp"

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
TEUCHOS_UNIT_TEST(TimeStepControlStrategyIntegralController, Default_Construction)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyIntegralController<double>());
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getController() != "PID");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKI() != 0.58);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKP() != 0.21);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKD() != 0.10);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactor() != 0.90);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactorAfterReject() != 0.90);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMax() != 5.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMin() != 0.5);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyIntegralController, Full_Construction)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyIntegralController<double>(
    "I", 0.6, 0.0, 0.0, 0.8, 0.8, 4.0, 0.4));
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getController() != "I");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKI() != 0.6);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKP() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKD() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactor() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactorAfterReject() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMax() != 4.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMin() != 0.4);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyIntegralController, Create_Constructor)
{
  // Construct ParmeterList for testing.
  auto tscs_temp =
    rcp(new Tempus::TimeStepControlStrategyIntegralController<double>(
      "I", 0.6, 0.0, 0.0, 0.8, 0.8, 4.0, 0.4));

  auto pl = rcp_const_cast<Teuchos::ParameterList>(tscs_temp->getValidParameters());

  auto tscs = Tempus::createTimeStepControlStrategyIntegralController<double>(pl);

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getController() != "I");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKI() != 0.6);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKP() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKD() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactor() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactorAfterReject() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMax() != 4.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMin() != 0.4);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyIntegralController, Accessors)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyIntegralController<double>());
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  tscs->setController("I");
  tscs->setKI(0.6);
  tscs->setKP(0.0);
  tscs->setKD(0.0);
  tscs->setSafetyFactor(0.8);
  tscs->setSafetyFactorAfterReject(0.8);
  tscs->setFacMax(4.0);
  tscs->setFacMin(0.4);

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getController() != "I");
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKI() != 0.6);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKP() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getKD() != 0.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactor() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getSafetyFactorAfterReject() != 0.8);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMax() != 4.0);
  TEUCHOS_TEST_FOR_EXCEPT(tscs->getFacMin() != 0.4);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyIntegralController, setNextTimeStep)
{
  double KI = 0.5;
  double KP = 0.25;
  double KD = 0.15;
  double safetyFactor = 0.9;
  double safetyFactorAfterReject = 0.9;
  double facMax = 5.0;
  double facMin = 0.5;

  auto tscs =
    rcp(new Tempus::TimeStepControlStrategyIntegralController<double>(
      "PID", KI, KP, KD, safetyFactor, safetyFactorAfterReject,
      facMax, facMin));

  // Setup the TimeStepControl --------------------------------
  auto tsc = rcp(new Tempus::TimeStepControl<double>());
  tsc->setTimeStepControlStrategy(tscs);
  tsc->setInitTime(0.0);
  tsc->setFinalTime(10.0);
  tsc->setMinTimeStep (0.01);
  tsc->setInitTimeStep(1.0);
  tsc->setMaxTimeStep (10.0);
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

  double order = 2.0;
  solutionHistory->addState(icState);
  solutionHistory->getCurrentState()->setTimeStep(1.0);
  solutionHistory->getCurrentState()->setTime(0.0);
  solutionHistory->getCurrentState()->setIndex(0);
  solutionHistory->getCurrentState()->setOrder(order);

  // Mock Integrator

  // -- First Time Step
  solutionHistory->initWorkingState();
  auto currentState = solutionHistory->getCurrentState();
  auto workingState = solutionHistory->getWorkingState();

  tsc->setNextTimeStep(solutionHistory, status);

  // First time step should cause no change to dt because
  // internal relative errors = 1.
  TEST_FLOATING_EQUALITY(workingState->getTimeStep(), 1.0, 1.0e-14);

  // Mock takeStep
  double errN = 0.1;
  workingState->setErrorRel(errN);
  workingState->setSolutionStatus(Tempus::Status::PASSED);



  // -- Second Time Step
  solutionHistory->initWorkingState();
  currentState = solutionHistory->getCurrentState();
  workingState = solutionHistory->getWorkingState();
  double dt = workingState->getTimeStep();

  tsc->setNextTimeStep(solutionHistory, status);

  double p    = order - 1.0;
  double dtNew = dt*safetyFactor*std::pow(errN, -KI/p);
  TEST_FLOATING_EQUALITY(workingState->getTimeStep(), dtNew, 1.0e-14);

  // Mock takeStep
  errN = 0.2;
  double errNm1 = 0.1;
  workingState->setErrorRel(errN);
  workingState->setSolutionStatus(Tempus::Status::PASSED);



  // -- Third Time Step
  solutionHistory->initWorkingState();
  currentState = solutionHistory->getCurrentState();
  workingState = solutionHistory->getWorkingState();
  dt = workingState->getTimeStep();

  tsc->setNextTimeStep(solutionHistory, status);

  dtNew = dt*safetyFactor*std::pow(errN,   -KI/p)
                         *std::pow(errNm1,  KP/p);
  TEST_FLOATING_EQUALITY(workingState->getTimeStep(), dtNew, 1.0e-14);

  // Mock takeStep
  errN = 0.3;
  errNm1 = 0.2;
  double errNm2 = 0.1;
  workingState->setErrorRel(errN);
  workingState->setSolutionStatus(Tempus::Status::PASSED);



  // -- Fourth Time Step
  solutionHistory->initWorkingState();
  currentState = solutionHistory->getCurrentState();
  workingState = solutionHistory->getWorkingState();
  dt = workingState->getTimeStep();

  tsc->setNextTimeStep(solutionHistory, status);

  dtNew = dt*safetyFactor*std::pow(errN,   -KI/p)
                         *std::pow(errNm1,  KP/p)
                         *std::pow(errNm2, -KD/p);
  TEST_FLOATING_EQUALITY(workingState->getTimeStep(), dtNew, 1.0e-14);

}


} // namespace Tempus_Test
