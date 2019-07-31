// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_RKButcherTableauFactory_hpp
#define Tempus_RKButcherTableauFactory_hpp

#include "Tempus_RKButcherTableau.hpp"
#include "Tempus_StepperRKButcherTableau.hpp"

namespace Tempus {


/** \brief Runge-Kutta factory class.
 *  This factory creates RKButcherTableau objects given the description
 *  string from the RKButcherTableau object.
 */
template<class Scalar>
class RKButcherTableauFactory
{
public:

  /// Constructor
  RKButcherTableauFactory(){}

  /// Destructor
  virtual ~RKButcherTableauFactory() {}

  /// Create tableau from stepper type (e.g., "RK Explicit 4 Stage").
  Teuchos::RCP<RKButcherTableau<Scalar> > createTableau(
    std::string stepperType = "RK Explicit 4 Stage")
  {
    Teuchos::RCP<RKButcherTableau<Scalar> > tableau =
      createTableau(stepperType, Teuchos::null);

    return tableau;
  }

  /// Create tableau from ParameterList.
  Teuchos::RCP<RKButcherTableau<Scalar> > createTableau(
    Teuchos::RCP<Teuchos::ParameterList> pl)
  {
    std::string stepperType =
      pl->get<std::string>("Stepper Type", "RK Explicit 4 Stage");
    Teuchos::RCP<RKButcherTableau<Scalar> > tableau =
      createTableau(stepperType, pl);

    return tableau;
  }

  /// Create tableau from its pieces.
  Teuchos::RCP<RKButcherTableau<Scalar> > createTableau(
    const Teuchos::SerialDenseMatrix<int,Scalar>& A,
    const Teuchos::SerialDenseVector<int,Scalar>& b,
    const Teuchos::SerialDenseVector<int,Scalar>& c,
    const int order,
    const int orderMin,
    const int orderMax,
    const std::string& longDescription,
    bool isEmbedded = false,
    const Teuchos::SerialDenseVector<int,Scalar>&
      bstar =  Teuchos::SerialDenseVector<int,Scalar>())
  {
    Teuchos::RCP<RKButcherTableau<Scalar> > tableau =
      rcp(new RKButcherTableau<Scalar>());

    tableau->initialize(A, b, c, order, orderMin, orderMax,
                       longDescription,isEmbedded,bstar);
    tableau->setPLTableau();

    return tableau;
  }


private:

  /// Create tableau.
  // stepperType should be consistent with the ParameterList.
  Teuchos::RCP<RKButcherTableau<Scalar> > createTableau(
    std::string stepperType, Teuchos::RCP<Teuchos::ParameterList> pl)
  {
    Teuchos::RCP<RKButcherTableau<Scalar> > tableau;

    // Very simple factory method

    // ERK
    if      (stepperType == "General ERK")                                        tableau = rcp(new GeneralExplicit_RKBT<Scalar>());
    else if (stepperType == "RK Forward Euler")                                   tableau = rcp(new ForwardEuler_RKBT<Scalar>());
    else if (stepperType == "RK Explicit Midpoint")                               tableau = rcp(new ExplicitMidpoint_RKBT<Scalar>());
    else if (stepperType == "RK Explicit Trapezoidal")                            tableau = rcp(new ExplicitTrapezoidal_RKBT<Scalar>());
    else if (stepperType == "Heuns Method")                                       tableau = rcp(new ExplicitTrapezoidal_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 3 Stage 3rd order")                      tableau = rcp(new Explicit3Stage3rdOrder_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 3 Stage 3rd order by Heun")              tableau = rcp(new Explicit3Stage3rdOrderHeun_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 3 Stage 3rd order TVD")                  tableau = rcp(new Explicit3Stage3rdOrderTVD_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 4 Stage 3rd order by Runge")             tableau = rcp(new Explicit4Stage3rdOrderRunge_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 5 Stage 3rd order by Kinnmark and Gray") tableau = rcp(new Explicit5Stage3rdOrderKandG_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 4 Stage")                                tableau = rcp(new Explicit4Stage4thOrder_RKBT<Scalar>());
    else if (stepperType == "RK Explicit 3/8 Rule")                               tableau = rcp(new Explicit3_8Rule_RKBT<Scalar>());
    else if (stepperType == "Bogacki-Shampine 3(2) Pair")                         tableau = rcp(new ExplicitBogackiShampine32_RKBT<Scalar>());
    else if (stepperType == "Merson 4(5) Pair")                                   tableau = rcp(new ExplicitMerson45_RKBT<Scalar>());

    // DIRK
    else if (stepperType == "General DIRK")             tableau = rcp(new GeneralDIRK_RKBT<Scalar>());
    else if (stepperType == "RK Backward Euler")        tableau = rcp(new BackwardEuler_RKBT<Scalar>());
    else if (stepperType == "IRK 1 Stage Theta Method") tableau = rcp(new IRK1StageTheta_RKBT<Scalar>());
    else if (stepperType == "RK Implicit Midpoint")     tableau = rcp(new ImplicitMidpoint_RKBT<Scalar>());
    else if (stepperType == "SDIRK 2 Stage 2nd order")  tableau = rcp(new SDIRK2Stage2ndOrder_RKBT<Scalar>());
    else if (stepperType == "SDIRK 2 Stage 3rd order")  tableau = rcp(new SDIRK2Stage3rdOrder_RKBT<Scalar>());
    else if (stepperType == "SDIRK 3 Stage 4th order")  tableau = rcp(new SDIRK3Stage4thOrder_RKBT<Scalar>());
    else if (stepperType == "SDIRK 5 Stage 4th order")  tableau = rcp(new SDIRK5Stage4thOrder_RKBT<Scalar>());
    else if (stepperType == "SDIRK 5 Stage 5th order")  tableau = rcp(new SDIRK5Stage5thOrder_RKBT<Scalar>());
    else if (stepperType == "SDIRK 2(1) Pair")          tableau = rcp(new SDIRK21_RKBT<Scalar>());

    // EDIRK
    else if (stepperType == "EDIRK 2 Stage 3rd order")    tableau = rcp(new EDIRK2Stage3rdOrder_RKBT<Scalar>());
    else if (stepperType == "RK Trapezoidal Rule")        tableau = rcp(new TrapezoidalRule_RKBT<Scalar>());
    else if (stepperType == "RK Crank-Nicolson")          tableau = rcp(new TrapezoidalRule_RKBT<Scalar>());
    else if (stepperType == "EDIRK 2 Stage Theta Method") tableau = rcp(new EDIRK2StageTheta_RKBT<Scalar>());

    // IRK
    else if (stepperType == "RK Implicit 2 Stage 4th order Gauss")                  tableau = rcp(new Implicit2Stage4thOrderGauss_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 6th order Gauss")                  tableau = rcp(new Implicit3Stage6thOrderGauss_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 4th Order Hammer & Hollingsworth") tableau = rcp(new Implicit2Stage4thOrderHammerHollingsworth_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 6th Order Kuntzmann & Butcher")    tableau = rcp(new Implicit3Stage6thOrderKuntzmannButcher_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 4 Stage 8th Order Kuntzmann & Butcher")    tableau = rcp(new Implicit4Stage8thOrderKuntzmannButcher_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 1 Stage 1st order Radau left")             tableau = rcp(new Implicit1Stage1stOrderRadauA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 3rd order Radau left")             tableau = rcp(new Implicit2Stage3rdOrderRadauA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 5th order Radau left")             tableau = rcp(new Implicit3Stage5thOrderRadauA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 1 Stage 1st order Radau right")            tableau = rcp(new Implicit1Stage1stOrderRadauB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 3rd order Radau right")            tableau = rcp(new Implicit2Stage3rdOrderRadauB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 5th order Radau right")            tableau = rcp(new Implicit3Stage5thOrderRadauB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 2nd order Lobatto A")              tableau = rcp(new Implicit2Stage2ndOrderLobattoA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 4th order Lobatto A")              tableau = rcp(new Implicit3Stage4thOrderLobattoA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 4 Stage 6th order Lobatto A")              tableau = rcp(new Implicit4Stage6thOrderLobattoA_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 2nd order Lobatto B")              tableau = rcp(new Implicit2Stage2ndOrderLobattoB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 4th order Lobatto B")              tableau = rcp(new Implicit3Stage4thOrderLobattoB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 4 Stage 6th order Lobatto B")              tableau = rcp(new Implicit4Stage6thOrderLobattoB_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 2 Stage 2nd order Lobatto C")              tableau = rcp(new Implicit2Stage2ndOrderLobattoC_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 3 Stage 4th order Lobatto C")              tableau = rcp(new Implicit3Stage4thOrderLobattoC_RKBT<Scalar>());
    else if (stepperType == "RK Implicit 4 Stage 6th order Lobatto C")              tableau = rcp(new Implicit4Stage6thOrderLobattoC_RKBT<Scalar>());
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown 'Stepper Type' = '" << stepperType << "'\n");
    }

    tableau->setParameterList(pl);

    return tableau;
  }

};


/// Nonmember constructor for RKButcherTableauFactory.
template<class Scalar>
Teuchos::RCP<RKButcherTableauFactory<Scalar> > RKBT_Factory()
{
  Teuchos::RCP<RKButcherTableauFactory<Scalar> > tf =
    Teuchos::rcp(new RKButcherTableauFactory<Scalar>());
  return tf;
}

/// Nonmember constructor for Butcher Tableau.
template<class Scalar>
Teuchos::RCP<RKButcherTableau<Scalar> > createRKButcherTableau(
  std::string stepperType = "RK Explicit 4 Stage")
{
  Teuchos::RCP<RKButcherTableauFactory<Scalar> > tf = RKBT_Factory<Scalar>();
  Teuchos::RCP<RKButcherTableau<Scalar> > tableau =
    tf->createTableau(stepperType);
  return tableau;
}

/// Nonmember constructor for Butcher Tableau.
template<class Scalar>
Teuchos::RCP<RKButcherTableau<Scalar> > createRKButcherTableau(
  Teuchos::RCP<Teuchos::ParameterList> pl)
{
  Teuchos::RCP<RKButcherTableauFactory<Scalar> > tf = RKBT_Factory<Scalar>();
  Teuchos::RCP<RKButcherTableau<Scalar> > tableau = tf->createTableau(pl);
  return tableau;
}

/// Nonmember constructor for Butcher Tableau.
template<class Scalar>
Teuchos::RCP<RKButcherTableau<Scalar> > createRKButcherTableau(
  const Teuchos::SerialDenseMatrix<int,Scalar>& A,
  const Teuchos::SerialDenseVector<int,Scalar>& b,
  const Teuchos::SerialDenseVector<int,Scalar>& c,
  const int order,
  const int orderMin,
  const int orderMax,
  const std::string& longDescription,
  bool isEmbedded = false,
  const Teuchos::SerialDenseVector<int,Scalar>&
    bstar =  Teuchos::SerialDenseVector<int,Scalar>())
{
  Teuchos::RCP<RKButcherTableauFactory<Scalar> > tf = RKBT_Factory<Scalar>();
  Teuchos::RCP<RKButcherTableau<Scalar> > tableau =
    tf->createTableau(A, b, c, order, orderMin, orderMax,
                     longDescription,isEmbedded,bstar);
  return tableau;
}


} // namespace Tempus
#endif // Tempus_RKButcherTableauFactory_hpp
