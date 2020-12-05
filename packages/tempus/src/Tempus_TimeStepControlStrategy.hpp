// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_TimeStepControlStrategy_hpp
#define Tempus_TimeStepControlStrategy_hpp

// Teuchos
#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"

#include "Tempus_SolutionHistory.hpp"


namespace Tempus {

template<class Scalar> class TimeStepControl;

/** \brief TimeStepControlStrategy class for TimeStepControl
 *
 *  This is the base class for TimeStepControlStrategies.
 *  The primary function required from derived classes is setNextTimeStep(),
 *  which will
 *   - determine the next step from information in the TimeStepControl
 *     and SolutionHistory (i.e., SolutionStates)
 *   - set the next time step on the workingState in the SolutionHistory
 *  If a valid timestep can not be determined the Status is set to FAILED.
 */
template<class Scalar>
class TimeStepControlStrategy
  : virtual public Teuchos::Describable,
    virtual public Teuchos::VerboseObject<Tempus::TimeStepControlStrategy<Scalar> >
{
public:

  /// Constructor
  TimeStepControlStrategy()
   : stepType_("Variable"), isInitialized_(false)
  {}

  /// Destructor
  virtual ~TimeStepControlStrategy(){}

  virtual std::string getStepType() const { return stepType_; }

#ifndef TEMPUS_HIDE_DEPRECATED_CODE
  /// Deprecated get the time step size.
  virtual void getNextTimeStep(
    const TimeStepControl<Scalar> tsc,
    Teuchos::RCP<SolutionHistory<Scalar> > sh,
    Status & integratorStatus)
  {
    this->setNextTimeStep(tsc, sh, integratorStatus);
  };
#endif

  /// Set the time step size.
  virtual void setNextTimeStep(
    const TimeStepControl<Scalar> & /* tsc */,
    Teuchos::RCP<SolutionHistory<Scalar> > /* sh */,
    Status & /* integratorStatus */) {}

  virtual void initialize() const { isInitialized_ = true; }
  virtual bool isInitialized() { return isInitialized_; }
  virtual void checkInitialized()
  {
    if ( !isInitialized_ ) {
      this->describe( *(this->getOStream()), Teuchos::VERB_MEDIUM);
      TEUCHOS_TEST_FOR_EXCEPTION( !isInitialized_, std::logic_error,
        "Error - " << this->description() << " is not initialized!");
    }
  }

protected:

  virtual void setStepType(std::string s) { stepType_ = s; }

  std::string  stepType_;       ///< Strategy Step Type, e.g., "Constant"
  mutable bool isInitialized_;  ///< Bool if TimeStepControl is initialized.

};


} // namespace Tempus
#endif // Tempus_TimeStepControlStrategy_hpp
