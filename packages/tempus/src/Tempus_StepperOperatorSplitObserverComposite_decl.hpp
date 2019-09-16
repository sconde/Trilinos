// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperOperatorSplitObserverComposite_hpp
#define Tempus_StepperOperatorSplitObserverComposite_hpp

#include "Tempus_SolutionHistory.hpp"
#include <vector>


namespace Tempus {

// Forward Declaration for recursive includes (this Observer <--> Stepper)
template<class Scalar> class StepperOperatorSplit;

/** \brief StepperOperatorSplitObserverComposite class for StepperOperatorSplit.
 *
 * This is a means for application developers to perform tasks
 * during the time steps, e.g.,
 *   - Compute specific quantities
 *   - Output information
 *   - "Massage" the working solution state
 *   - ...
 *
 * <b>Design Considerations</b>
 *   - StepperOperatorSplitObserverComposite is not stateless!  Developers may touch the
 *     solution state!  Developers need to be careful not to break the
 *     restart (checkpoint) capability.
 */
template<class Scalar>
class StepperOperatorSplitObserverComposite
 : virtual public Tempus::StepperObserver<Scalar>
{
public:

  /// Constructor
  StepperOperatorSplitObserverComposite(){}

  /// Destructor
  virtual ~StepperOperatorSplitObserverComposite(){}

  /// Observe Stepper at beginning of takeStep.
  virtual void observeBeginTakeStep(
    Teuchos::RCP<SolutionHistory<Scalar> > /* sh */,
    Stepper<Scalar> & /* stepper */) override;

  /// Observe Stepper before index subStepper->takeStep()
  virtual void observeBeforeStepper(int /* index */,
    Teuchos::RCP<SolutionHistory<Scalar> > /* sh */,
    StepperOperatorSplit<Scalar> & /* stepperOS */) override;

  /// Observe Stepper after index subStepper->takeStep()
  virtual void observeAfterStepper(int /* index */,
    Teuchos::RCP<SolutionHistory<Scalar> > /* sh */,
    StepperOperatorSplit<Scalar> & /* stepperOS */) override;

  /// Observe Stepper at end of takeStep.
  virtual void observeEndTakeStep(
    Teuchos::RCP<SolutionHistory<Scalar> > /* sh */,
    Stepper<Scalar> & /* stepperOS */) override;
private:
  std::vector<Teuchos::RCP<StepperOperatorSplitObserver<Scalar> > > observers_;

};
} // namespace Tempus
#endif // Tempus_StepperOperatorSplitObserverComposite_hpp
