// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Tempus_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_TEMPUS_EXPLICIT_INSTANTIATION
#include "Tempus_Stepper.hpp"
#include "Tempus_Stepper_impl.hpp"

namespace Tempus {

  TEMPUS_INSTANTIATE_TEMPLATE_CLASS(Stepper)

  // Provide basic parameters to Steppers.
  void getValidParametersBasic(
    Teuchos::RCP<Teuchos::ParameterList> pl, std::string stepperType);

  // Returns the default solver ParameterList for implicit Steppers.
  Teuchos::RCP<Teuchos::ParameterList> defaultSolverParameters();

}

#endif
