// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperFactory_hpp
#define Tempus_StepperFactory_hpp

#include "Teuchos_ParameterList.hpp"
#include "Tempus_StepperForwardEuler.hpp"
#include "Tempus_StepperBackwardEuler.hpp"
#include "Tempus_StepperBDF2.hpp"
#include "Tempus_StepperNewmarkImplicitAForm.hpp"
#include "Tempus_StepperNewmarkImplicitDForm.hpp"
#include "Tempus_StepperNewmarkExplicitAForm.hpp"
#include "Tempus_StepperHHTAlpha.hpp"
#include "Tempus_StepperExplicitRK.hpp"
#include "Tempus_StepperRKButcherTableau.hpp"
#include "Tempus_StepperDIRK.hpp"
#include "Tempus_StepperIMEX_RK.hpp"
#include "Tempus_StepperIMEX_RK_Partition.hpp"
#include "Tempus_StepperLeapfrog.hpp"
#include "Tempus_StepperOperatorSplit.hpp"
#include "Tempus_StepperTrapezoidal.hpp"


namespace Tempus {

/** \brief Stepper factory.
 *
 * <b>Adding Steppers</b>
 *    -#
 */
template<class Scalar>
class StepperFactory
{
public:

  /// Constructor
  StepperFactory(){}

  /// Destructor
  virtual ~StepperFactory() {}

  /// Create default stepper from stepper type (e.g., "Forward Euler").
  Teuchos::RCP<Stepper<Scalar> > createStepper(
    std::string stepperType = "Forward Euler",
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >&
      model = Teuchos::null)
  {
    if (stepperType == "") stepperType = "Forward Euler";
    return this->createStepper(model, stepperType, Teuchos::null);
  }

  /// Create stepper from ParameterList with its details.
  Teuchos::RCP<Stepper<Scalar> > createStepper(
    Teuchos::RCP<Teuchos::ParameterList> stepperPL,
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >&
      model = Teuchos::null)
  {
    std::string stepperType = "Forward Euler";
    if (stepperPL != Teuchos::null)
      stepperType = stepperPL->get<std::string>("Stepper Type","Forward Euler");
    return this->createStepper(model, stepperType, stepperPL);
  }

  /// Create stepper from ParameterList with its details.
  Teuchos::RCP<Stepper<Scalar> > createStepper(
    Teuchos::RCP<Teuchos::ParameterList> stepperPL,
    std::vector<Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> > >
      models = Teuchos::null)
  {
    std::string stepperType = stepperPL->get<std::string>("Stepper Type");
    return this->createStepper(models, stepperType, stepperPL);
  }


  // ---------------------------------------------------------------------------

  void setStepperValuesFromPL(
    Teuchos::RCP<Stepper<Scalar> > stepper,
    Teuchos::RCP<Teuchos::ParameterList> stepperPL)
  {
    auto stepperType =
      stepperPL->get<std::string>("Stepper Type", stepper->description());
    TEUCHOS_TEST_FOR_EXCEPTION(
      stepperType != stepper->description() ,std::runtime_error,
      "  ParameterList 'Stepper Type' (='" + stepperType +"')\n"
      "  does not match type for stepper Stepper (='"
      + stepper->description() + "').");

    stepper->setUseFSAL(
      stepperPL->get<bool>("Use FSAL", stepper->getUseFSALDefault()));

    stepper->setICConsistency(
      stepperPL->get<std::string>("Initial Condition Consistency",
                                  stepper->getICConsistencyDefault()));

    stepper->setICConsistencyCheck(
      stepperPL->get<bool>("Initial Condition Consistency Check",
                           stepper->getICConsistencyCheckDefault()));
  }


  void setGeneralTableauFromPL(
    Teuchos::RCP<StepperGeneralERK<Scalar> > stepper,
    Teuchos::RCP<Teuchos::ParameterList> stepperPL)
  {
    using Teuchos::as;
    using Teuchos::RCP;
    using Teuchos::rcp_const_cast;
    using Teuchos::ParameterList;

    RCP<ParameterList> tableauPL = sublist(stepperPL,"Tableau",true);
    std::size_t numStages = 0;
    int order = tableauPL->get<int>("order");
    Teuchos::SerialDenseMatrix<int,Scalar> A;
    Teuchos::SerialDenseVector<int,Scalar> b;
    Teuchos::SerialDenseVector<int,Scalar> c;
    Teuchos::SerialDenseVector<int,Scalar> bstar;

    // read in the A matrix
    {
      std::vector<std::string> A_row_tokens;
      Tempus::StringTokenizer(A_row_tokens, tableauPL->get<std::string>("A"),
                              ";",true);

      // this is the only place where numStages is set
      numStages = A_row_tokens.size();

      // allocate the matrix
      A.shape(as<int>(numStages),as<int>(numStages));

      // fill the rows
      for(std::size_t row=0;row<numStages;row++) {
        // parse the row (tokenize on space)
        std::vector<std::string> tokens;
        Tempus::StringTokenizer(tokens,A_row_tokens[row]," ",true);

        std::vector<double> values;
        Tempus::TokensToDoubles(values,tokens);

        TEUCHOS_TEST_FOR_EXCEPTION(values.size()!=numStages,std::runtime_error,
          "Error parsing A matrix, wrong number of stages in row "
          << row << "\n" + stepper->description());

        for(std::size_t col=0;col<numStages;col++)
          A(row,col) = values[col];
      }
    }

    // size b and c vectors
    b.size(as<int>(numStages));
    c.size(as<int>(numStages));

    // read in the b vector
    {
      std::vector<std::string> tokens;
      Tempus::StringTokenizer(tokens,tableauPL->get<std::string>("b")," ",true);
      std::vector<double> values;
      Tempus::TokensToDoubles(values,tokens);

      TEUCHOS_TEST_FOR_EXCEPTION(values.size()!=numStages,std::runtime_error,
        "Error parsing b vector, wrong number of stages.\n"
        + stepper->description());

      for(std::size_t i=0;i<numStages;i++)
        b(i) = values[i];
    }

    // read in the c vector
    {
      std::vector<std::string> tokens;
      Tempus::StringTokenizer(tokens,tableauPL->get<std::string>("c")," ",true);
      std::vector<double> values;
      Tempus::TokensToDoubles(values,tokens);

      TEUCHOS_TEST_FOR_EXCEPTION(values.size()!=numStages,std::runtime_error,
        "Error parsing c vector, wrong number of stages.\n"
        + stepper->description());

      for(std::size_t i=0;i<numStages;i++)
        c(i) = values[i];
    }

    if (tableauPL->isParameter("bstar") and
        tableauPL->get<std::string>("bstar") != "") {
      bstar.size(as<int>(numStages));
      // read in the bstar vector
      {
        std::vector<std::string> tokens;
        Tempus::StringTokenizer(
          tokens, tableauPL->get<std::string>("bstar"), " ", true);
        std::vector<double> values;
        Tempus::TokensToDoubles(values,tokens);

        TEUCHOS_TEST_FOR_EXCEPTION(values.size()!=numStages,std::runtime_error,
          "Error parsing bstar vector, wrong number of stages.\n"
          "      Number of RK stages    = " << numStages << "\n"
          "      Number of bstar values = " << values.size() << "\n"
          + stepper->description());

        for(std::size_t i=0;i<numStages;i++)
          bstar(i) = values[i];
      }
        stepper->setTableau(A,b,c,order,order,order,bstar);
    } else {
        stepper->setTableau(A,b,c,order,order,order);
    }
  }


  Teuchos::RCP<StepperGeneralERK<Scalar> > createStepperGeneralERK(
      const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& model,
      Teuchos::RCP<Teuchos::ParameterList> stepperPL)
  {
    Teuchos::RCP<StepperGeneralERK<Scalar> > stepper =
      Teuchos::rcp(new StepperGeneralERK<Scalar>());

    if (stepperPL != Teuchos::null) {

      stepperPL->validateParametersAndSetDefaults(
                                              *stepper->getValidParameters());

      setStepperValuesFromPL(stepper, stepperPL);

      stepper->setUseEmbedded(
        stepperPL->get<bool>("Use Embedded",stepper->getUseEmbeddedDefault()));

      setGeneralTableauFromPL(stepper, stepperPL);
      TEUCHOS_TEST_FOR_EXCEPTION(stepper->isImplicit() == true,std::logic_error,
        "Error - General ERK received an implicit Butcher Tableau!\n");
    }

    if (model != Teuchos::null) {
      stepper->setModel(model);
      stepper->initialize();
    }

    return stepper;
  }

  // ---------------------------------------------------------------------------

private:

  /// Very simple factory method
  Teuchos::RCP<Stepper<Scalar> > createStepper(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& model,
    std::string stepperType,
    Teuchos::RCP<Teuchos::ParameterList> stepperPL)
  {
    using Teuchos::rcp;
    if (stepperType == "Forward Euler")
      return rcp(new StepperForwardEuler<Scalar>(model, stepperPL));
    else if (stepperType == "Backward Euler")
      return rcp(new StepperBackwardEuler<Scalar>(model, stepperPL));
    else if (stepperType == "Trapezoidal Method")
      return rcp(new StepperTrapezoidal<Scalar>(model, stepperPL));
    else if (stepperType == "BDF2")
      return rcp(new StepperBDF2<Scalar>(model, stepperPL));
    else if (stepperType == "Newmark Implicit a-Form")
      return rcp(new StepperNewmarkImplicitAForm<Scalar>(model, stepperPL));
    else if (stepperType == "Newmark Implicit d-Form")
      return rcp(new StepperNewmarkImplicitDForm<Scalar>(model, stepperPL));
    else if (stepperType == "Newmark Explicit a-Form")
      return rcp(new StepperNewmarkExplicitAForm<Scalar>(model, stepperPL));
    else if (stepperType == "HHT-Alpha")
      return rcp(new StepperHHTAlpha<Scalar>(model, stepperPL));
    else if (stepperType == "General ERK" )
      return createStepperGeneralERK(model, stepperPL);
    else if (
      stepperType == "RK Forward Euler" ||
      stepperType == "RK Explicit 4 Stage" ||
      stepperType == "RK Explicit 3/8 Rule" ||
      stepperType == "RK Explicit 4 Stage 3rd order by Runge" ||
      stepperType == "RK Explicit 5 Stage 3rd order by Kinnmark and Gray"||
      stepperType == "RK Explicit 3 Stage 3rd order" ||
      stepperType == "RK Explicit 3 Stage 3rd order TVD" ||
      stepperType == "RK Explicit 3 Stage 3rd order by Heun" ||
      stepperType == "RK Explicit Midpoint" ||
      stepperType == "RK Explicit Trapezoidal" || stepperType == "Heuns Method" ||
      stepperType == "Bogacki-Shampine 3(2) Pair" ||
      stepperType == "Merson 4(5) Pair")
      return rcp(new StepperExplicitRK<Scalar>(model, stepperType, stepperPL));
    else if (
      stepperType == "RK Backward Euler" ||
      stepperType == "IRK 1 Stage Theta Method" ||
      stepperType == "RK Implicit Midpoint" ||
      stepperType == "SDIRK 1 Stage 1st order" ||
      stepperType == "SDIRK 2 Stage 2nd order" ||
      stepperType == "SDIRK 2 Stage 3rd order" ||
      stepperType == "EDIRK 2 Stage 3rd order" ||
      stepperType == "EDIRK 2 Stage Theta Method" ||
      stepperType == "SDIRK 3 Stage 4th order" ||
      stepperType == "SDIRK 5 Stage 4th order" ||
      stepperType == "SDIRK 5 Stage 5th order" ||
      stepperType == "SDIRK 2(1) Pair" ||
      stepperType == "RK Trapezoidal Rule" || stepperType == "RK Crank-Nicolson" ||
      stepperType == "General DIRK"
      )
      return rcp(new StepperDIRK<Scalar>(model, stepperType, stepperPL));
    else if (
      stepperType == "RK Implicit 3 Stage 6th Order Kuntzmann & Butcher" ||
      stepperType == "RK Implicit 4 Stage 8th Order Kuntzmann & Butcher" ||
      stepperType == "RK Implicit 2 Stage 4th Order Hammer & Hollingsworth" ||
      stepperType == "RK Implicit 1 Stage 2nd order Gauss" ||
      stepperType == "RK Implicit 2 Stage 4th order Gauss" ||
      stepperType == "RK Implicit 3 Stage 6th order Gauss" ||
      stepperType == "RK Implicit 1 Stage 1st order Radau left" ||
      stepperType == "RK Implicit 2 Stage 3rd order Radau left" ||
      stepperType == "RK Implicit 3 Stage 5th order Radau left" ||
      stepperType == "RK Implicit 1 Stage 1st order Radau right" ||
      stepperType == "RK Implicit 2 Stage 3rd order Radau right" ||
      stepperType == "RK Implicit 3 Stage 5th order Radau right" ||
      stepperType == "RK Implicit 2 Stage 2nd order Lobatto A" ||
      stepperType == "RK Implicit 3 Stage 4th order Lobatto A" ||
      stepperType == "RK Implicit 4 Stage 6th order Lobatto A" ||
      stepperType == "RK Implicit 2 Stage 2nd order Lobatto B" ||
      stepperType == "RK Implicit 3 Stage 4th order Lobatto B" ||
      stepperType == "RK Implicit 4 Stage 6th order Lobatto B" ||
      stepperType == "RK Implicit 2 Stage 2nd order Lobatto C" ||
      stepperType == "RK Implicit 3 Stage 4th order Lobatto C" ||
      stepperType == "RK Implicit 4 Stage 6th order Lobatto C" ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - Implicit RK not implemented yet!.\n");
    }
    else if (
      stepperType == "IMEX RK 1st order" ||
      stepperType == "IMEX RK SSP2"      ||
      stepperType == "IMEX RK ARS 233"   ||
      stepperType == "General IMEX RK" )
      return rcp(new StepperIMEX_RK<Scalar>(model, stepperType, stepperPL));
    else if (
      stepperType == "Partitioned IMEX RK 1st order" ||
      stepperType == "Partitioned IMEX RK SSP2"      ||
      stepperType == "Partitioned IMEX RK ARS 233"   ||
      stepperType == "General Partitioned IMEX RK" )
      return rcp(new StepperIMEX_RK_Partition<Scalar>(
                        model, stepperType, stepperPL));
    else if (stepperType == "Leapfrog")
      return rcp(new StepperLeapfrog<Scalar>(model, stepperPL));
    else {
      Teuchos::RCP<Teuchos::FancyOStream> out =
        Teuchos::VerboseObjectBase::getDefaultOStream();
      Teuchos::OSTab ostab(out,1,"StepperFactory::createStepper");
      *out
      << "Unknown Stepper Type!  ('"+stepperType+"').\n"
      << "Here is a list of available Steppers.\n"
      << "  One-Step Methods:\n"
      << "    'Forward Euler'\n"
      << "    'Backward Euler'\n"
      << "    'Trapezoidal Method'\n"
      << "  Multi-Step Methods:\n"
      << "    'BDF2'\n"
      << "  Second-order PDE Methods:\n"
      << "    'Leapfrog'\n"
      << "    'Newmark Implicit a-Form'\n"
      << "    'Newmark Implicit d-Form'\n"
      << "    'Newmark Explicit a-Form'\n"
      << "    'HHT-Alpha'\n"
      << "  Explicit Runge-Kutta Methods:\n"
      << "    'RK Forward Euler'\n"
      << "    'RK Explicit 4 Stage'\n"
      << "    'RK Explicit 3/8 Rule'\n"
      << "    'RK Explicit 4 Stage 3rd order by Runge'\n"
      << "    'RK Explicit 5 Stage 3rd order by Kinnmark and Gray'\n"
      << "    'RK Explicit 3 Stage 3rd order'\n"
      << "    'RK Explicit 3 Stage 3rd order TVD'\n"
      << "    'RK Explicit 3 Stage 3rd order by Heun'\n"
      << "    'RK Explicit Midpoint'\n"
      << "    'RK Explicit Trapezoidal' or 'Heuns Method'\n"
      << "    'Bogacki-Shampine 3(2) Pair'\n"
      << "    'General ERK'\n"
      << "  Implicit Runge-Kutta Methods:\n"
      << "    'RK Backward Euler'\n"
      << "    'IRK 1 Stage Theta Method'\n"
      << "    'RK Implicit Midpoint'\n"
      << "    'SDIRK 1 Stage 1st order'\n"
      << "    'SDIRK 2 Stage 2nd order'\n"
      << "    'SDIRK 2 Stage 3rd order'\n"
      << "    'EDIRK 2 Stage 3rd order'\n"
      << "    'EDIRK 2 Stage Theta Method'\n"
      << "    'SDIRK 3 Stage 4th order'\n"
      << "    'SDIRK 5 Stage 4th order'\n"
      << "    'SDIRK 5 Stage 5th order'\n"
      << "    'SDIRK 2(1) Pair'\n"
      << "    'RK Trapezoidal Rule' or 'RK Crank-Nicolson'\n"
      << "    'General DIRK'\n"
      << "  Implicit-Explicit (IMEX) Methods:\n"
      << "    'IMEX RK 1st order'\n"
      << "    'IMEX RK SSP2'\n"
      << "    'IMEX RK ARS 233'\n"
      << "    'General IMEX RK'\n"
      << "    'Partitioned IMEX RK 1st order'\n"
      << "    'Partitioned IMEX RK SSP2'\n"
      << "    'Partitioned IMEX RK ARS 233'\n"
      << "    'General Partitioned IMEX RK'\n"
      << "  Steppers with subSteppers:\n"
      << "    'Operator Split'\n"
      << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown 'Stepper Type' = " << stepperType);
    }
  }

  Teuchos::RCP<Stepper<Scalar> > createStepper(
    std::vector<Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> > > models,
    std::string stepperType,
    Teuchos::RCP<Teuchos::ParameterList> stepperPL)
  {
    if (stepperType == "Operator Split")
      return rcp(new StepperOperatorSplit<Scalar>(models, stepperPL));
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown 'Stepper Type' = " << stepperType);
    }
  }

};


} // namespace Tempus
#endif // Tempus_StepperFactory_hpp
