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

#include "Thyra_VectorStdOps.hpp"

#include "Tempus_IntegratorBasic.hpp"
#include "Tempus_StepperExplicitRK.hpp"

#include "../TestModels/BrusselatorModel.hpp"
#include "../TestUtils/Tempus_ConvergenceTestUtils.hpp"

#include <fstream>
#include <vector>

namespace Tempus_Test {

using Teuchos::RCP;
using Teuchos::ParameterList;
using Teuchos::sublist;
using Teuchos::getParametersFromXmlFile;

using Tempus::IntegratorBasic;
using Tempus::SolutionHistory;
using Tempus::SolutionState;

// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(EmbeddedPaper, Brusselator)
{
  std::vector<double> ErrorNorm;
  std::vector<double> tolRange;
  std::vector<int> VecAccept;
  std::vector<int> VecReject;
  std::vector<int> VecTotalStep;
  std::vector<int> VecWork;
  std::vector<double> VecAccuracy;
  tolRange.push_back(1e-2);
  tolRange.push_back(1e-3);
  tolRange.push_back(1e-4);
  tolRange.push_back(1e-5);
  tolRange.push_back(1e-6);
  tolRange.push_back(1e-7);
  //for (std::size_t n=0; n<tolRange.size();n++) {
  for (std::size_t n=0; n<1;n++) {
  
  
     const double tol = tolRange[n];

    // Read params from .xml file
    RCP<ParameterList> pList =
      getParametersFromXmlFile("Tempus_EmbeddedPaper_Brusselator.xml");

    // Setup the SinCosModel
    RCP<ParameterList> vdpm_pl = sublist(pList, "BrusselatorModel", true);
     RCP<BrusselatorModel<double> > model =
        Teuchos::rcp(new BrusselatorModel<double>(vdpm_pl));

    // Setup the Integrator and reset absolute/relative tolerance
    RCP<ParameterList> pl = sublist(pList, "Tempus", true);
    pl->sublist("Demo Integrator")
       .sublist("Time Step Control").set("Maximum Absolute Error", tol);
    pl->sublist("Demo Integrator")
       .sublist("Time Step Control").set("Maximum Relative Error", tol);

    const std::string stepperName = pl->sublist("Demo Integrator").get<std::string>("Stepper Name");

    volatile int numStage = -1;
    if (pl->sublist(stepperName).isParameter("Number of Stage") ){
       numStage = pl->sublist(stepperName).get<int>("Number of Stage");
       pl->sublist(stepperName).remove("Number of Stage");
    }

    // Initial Conditions
    // During the Integrator construction, the initial SolutionState
    // is set by default to model->getNominalVales().get_x().  However,
    // the application can set it also by integrator->setInitialState.
    RCP<Tempus::IntegratorBasic<double> > integrator =
      Tempus::integratorBasic<double>(pl, model);
    //order = integrator->getStepper()->getOrder();

     // Integrate to timeMax
    std::cout << "\nSIDAFA: integration started..." << std::endl;
    bool integratorStatus = integrator->advanceTime();
    TEST_ASSERT(integratorStatus);
    std::cout << "\nSIDAFA: integration done!" << std::endl;

     // Test if at 'Final Time'
     double time = integrator->getTime();
     double timeFinal = pl->sublist("Demo Integrator")
        .sublist("Time Step Control").get<double>("Final Time");
     TEST_FLOATING_EQUALITY(time, timeFinal, 1.0e-14);


     // Numerical reference solution at timeFinal (for \epsilon = 0.1)
     RCP<Thyra::VectorBase<double> > x = integrator->getX();
     RCP<Thyra::VectorBase<double> > xref = x->clone_v();
     Thyra::set_ele(0, 0.455808598718850, xref.ptr());
     Thyra::set_ele(1, 4.457846674977493, xref.ptr());

     // Calculate the error
     RCP<Thyra::VectorBase<double> > xdiff = x->clone_v();
     Thyra::V_StVpStV(xdiff.ptr(), 1.0, *xref, -1.0, *(x));
     const double L2norm = Thyra::norm_2(*xdiff);
     VecAccuracy.push_back(L2norm);

    // Plot sample solution and exact solution

    if (n == 0) {
      std::ofstream ftmp("Tempus_EmbeddedPaper_BrusselatorExample.dat");
      RCP<const SolutionHistory<double> > solutionHistory =
        integrator->getSolutionHistory();
      RCP<const Thyra::VectorBase<double> > x_exact_plot;
      RCP<const Thyra::VectorBase<double> > xDot_exact_plot;
      for (int i=0; i<solutionHistory->getNumStates(); i++) {
        RCP<const SolutionState<double> > solutionState = (*solutionHistory)[i];
        double time = solutionState->getTime();
        RCP<const Thyra::VectorBase<double> > x_plot = solutionState->getX();
        RCP<const Thyra::VectorBase<double> > xDot_plot = solutionState->getXDot();
        //x_exact_plot = model->getExactSolution(time).get_x();
        //xDot_exact_plot = model->getExactSolution(time).get_x_dot();
        ftmp << time << "   "
             << Thyra::get_ele(*(x_plot), 0) << "   "
             << Thyra::get_ele(*(x_plot), 1) << "   " << std::endl;
      }
      ftmp.close();
    }

     // get data
    const int nAccpt = integrator->getSolutionHistory()->
       getCurrentState()->getIndex();
    const int nFail = integrator->getSolutionHistory()->
       getCurrentState()->getMetaData()->getNRunningFailures();
    const int nSteps = nAccpt + nFail;
    const int FnEvalCnt = (numStage*nSteps) + 2;

    VecAccept.push_back(nAccpt);
    VecReject.push_back(nFail);
    VecTotalStep.push_back(nSteps);
    VecWork.push_back(FnEvalCnt);

  }

  std::cout << "Work-Precision Info:\n" << std::endl;
  for (std::size_t n=0; n<tolRange.size();n++) {
    // test for number of steps
    //TEST_EQUALITY(nAccpt, refIstep);
    std::cout << std::setw(4) << n << "   "
       << std::scientific << tolRange[n] << "   "
       << std::setw(6) << VecAccept[n] << "   "
       << std::setw(6) << VecReject[n] << "   "
       << std::setw(6) << VecTotalStep[n] << "   "
       << std::setw(6) << VecWork[n] << "   "
       << std::scientific << VecAccuracy[n] 
       << std::endl;
  }


  // Check the order and intercept
  //double slope = computeLinearRegressionLogLog<double>(StepSize, ErrorNorm);
  //std::cout << "  Stepper = EmbeddedPaper" << std::endl;
  //std::cout << "  =========================" << std::endl;
  //std::cout << "  Expected order: " << order << std::endl;
  //std::cout << "  Observed order: " << slope << std::endl;
  //std::cout << "  =========================" << std::endl;
  //TEST_FLOATING_EQUALITY( slope, order, 0.01 );
  ////TEST_FLOATING_EQUALITY( ErrorNorm[0], 0.051123, 1.0e-4 );

  //std::ofstream ftmp("Tempus_EmbeddedPaper_SinCos-Error.dat");
  //double error0 = 0.8*ErrorNorm[0];
  //for (int n=0; n<nTimeStepSizes; n++) {
    //ftmp << StepSize[n]  << "   " << ErrorNorm[n] << "   "
       //<< ErrorDotNorm[n] << std::endl;
  //}
  //ftmp.close();

  Teuchos::TimeMonitor::summarize();
}

} // namespace Tempus_Test
