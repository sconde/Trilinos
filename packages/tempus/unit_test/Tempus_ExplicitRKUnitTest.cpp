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
//#include "Tempus_StepperExplicitRK.hpp"

#include "../TestModels/SinCosModel.hpp"
#include "../TestModels/VanDerPolModel.hpp"
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

using Tempus::StepperFactory;
using Tempus::StepperExplicitRK_new;

// Comment out any of the following tests to exclude from build/run.
#define SETTABLEAU


#ifdef SETTABLEAU
// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ExplicitRKUnitTest, setTableau)
{
  RCP<StepperFactory<double> > sf = Teuchos::rcp(new StepperFactory<double>());
  RCP<StepperExplicitRK_new<double> > stepperRef;
  RCP<StepperExplicitRK_new<double> > stepperReset;

  // Setup the SinCosModel
  RCP<ParameterList> pList =
    getParametersFromXmlFile("../test/ExplicitRK/Tempus_ExplicitRK_SinCos.xml");
  RCP<ParameterList> scm_pl = sublist(pList, "SinCosModel", true);
  auto model = rcp(new Tempus_Test::SinCosModel<double>(scm_pl));

  std::vector<std::string> RKMethods;
  RKMethods.push_back("General ERK");
  RKMethods.push_back("RK Forward Euler");
  RKMethods.push_back("RK Explicit 4 Stage");
  RKMethods.push_back("RK Explicit 3/8 Rule");
  RKMethods.push_back("RK Explicit 4 Stage 3rd order by Runge");
  RKMethods.push_back("RK Explicit 5 Stage 3rd order by Kinnmark and Gray");
  RKMethods.push_back("RK Explicit 3 Stage 3rd order");
  RKMethods.push_back("RK Explicit 3 Stage 3rd order TVD");
  RKMethods.push_back("RK Explicit 3 Stage 3rd order by Heun");
  RKMethods.push_back("RK Explicit Midpoint");
  RKMethods.push_back("RK Explicit Trapezoidal");
  RKMethods.push_back("Heuns Method");
  RKMethods.push_back("Bogacki-Shampine 3(2) Pair");
  RKMethods.push_back("Merson 4(5) Pair");

  std::vector<std::string> resetMethod;
  resetMethod.push_back("stepperType");
  resetMethod.push_back("ParameterList");
  resetMethod.push_back("tableau");

//  for(std::vector<std::string>::size_type m = 0; m != RKMethods.size(); m++) {
//
//    //  Reference Stepper
//    std::string stepperType = RKMethods[m];
//    std::cout << "  a -  ExplicitRKUnitTest" << std::endl;
//    std::cout << " stepperRef = " << stepperRef << std::endl;
//    stepperRef = rcp_dynamic_cast<StepperExplicitRK_new<double> >(
//      sf->createStepper(stepperType, model));
//    std::cout << "  b -  ExplicitRKUnitTest" << std::endl;
//    std::cout << " stepperRef = " << stepperRef << std::endl;
//    auto plRef = stepperRef->getParameterList();
//    auto tableau = rcp_const_cast<Tempus::RKButcherTableau<double> >( stepperRef->getTableau());
//
//    for(std::vector<std::string>::size_type r = 0; r != resetMethod.size(); r++)
//    {
//      //  Reset Stepper
//      stepperReset = rcp_dynamic_cast<StepperExplicitRK_new<double> >(
//        sf->createStepper("RK Explicit 4 Stage", model));
//
//      if (resetMethod[r] == "stepperType") {          // Reset via stepperType
//        stepperReset->setTableau(stepperType);
//
//      } else if (resetMethod[r] == "ParameterList") { // Reset via ParameterList
//        RCP<ParameterList> pl = Teuchos::parameterList();
//        pl->setParameters(*plRef);
//        stepperReset->setTableauPL(pl);
//
//      } else if (resetMethod[r] == "tableau") {       // Reset via Tableau
//        stepperReset->setTableau(tableau);
//
//      } else {
//        std::cout << "Invalid reset method (" << resetMethod[r] << ").\n";
//        TEST_ASSERT(false)
//      }
//
//      stepperReset->initialize();
//      auto plReset = stepperReset->getParameterList();
//
//      bool pass = haveSameValues(*plReset, *plRef, true);
//      if (!pass) {
//        std::cout << std::endl;
//        std::cout << "----  Reset via stepperType -------------" << std::endl;
//        std::cout << "*** stepperType = " << RKMethods[m] << std::endl;
//        std::cout << "-----------------------------------------" << std::endl;
//        std::cout << "plRef   -------------- \n" << *plRef   << std::endl;
//        std::cout << "plReset -------------- \n" << *plReset << std::endl;
//      }
//      TEST_ASSERT(pass)
//    }
//
//  }
}
#endif // SETTABLEAU


} // namespace Tempus_Test
