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
#include "Tempus_TimeStepControlStrategyConstant.hpp"
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
TEUCHOS_UNIT_TEST(TimeStepControlStrategyConstant, Default_Construction)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyConstant<double>());
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  TEUCHOS_TEST_FOR_EXCEPT(tscs->getStepType() != "Constant");
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(TimeStepControlStrategyConstant, setNextTimeStep)
{
  auto tscs = rcp(new Tempus::TimeStepControlStrategyConstant<double>());
  TEUCHOS_TEST_FOR_EXCEPT(!tscs->isInitialized());

  double initTime  = 1.0;
  int    initIndex = -100;

  // Setup the SolutionHistory --------------------------------
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =model->getNominalValues();
  auto icSolution = rcp_const_cast<Thyra::VectorBase<double> >(inArgsIC.get_x());
  auto icState = Tempus::createSolutionStateX<double>(icSolution);
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());
  solutionHistory->addState(icState);
  solutionHistory->getCurrentState()->setTimeStep(0.9);
  solutionHistory->getCurrentState()->setTime(initTime);
  solutionHistory->getCurrentState()->setIndex(initIndex);

  // Setup the TimeStepControl --------------------------------
  auto tsc = rcp(new Tempus::TimeStepControl<double>());
  tsc->setTimeStepControlStrategy(tscs);
  tsc->setInitTime(initTime);
  tsc->setFinalTime(100.0);
  tsc->setMinTimeStep(0.01);
  tsc->setInitTimeStep(0.02);
  tsc->setMaxTimeStep(0.05);
  tsc->setInitIndex(initIndex);
  tsc->setFinalIndex(100);
  tsc->initialize();
  TEUCHOS_TEST_FOR_EXCEPT(!tsc->isInitialized());
  Tempus::Status status = Tempus::Status::WORKING;

  // ** Timestep ** //
  solutionHistory->initWorkingState();

  tsc->setNextTimeStep(solutionHistory, status);
  // ** Timestep ** //

  auto workingState = solutionHistory->getWorkingState();
  TEST_FLOATING_EQUALITY( workingState->getTimeStep(), 0.02, 1.0e-14);
  TEST_FLOATING_EQUALITY( workingState->getTime(), 1.02, 1.0e-14);
}


} // namespace Tempus_Test
