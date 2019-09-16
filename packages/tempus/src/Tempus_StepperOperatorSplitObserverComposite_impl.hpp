// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperOperatorSplitObserverComposite_impl_hpp
#define Tempus_StepperOperatorSplitObserverComposite_impl_hpp

#include "Tempus_SolutionHistory.hpp"


namespace Tempus {

template<class Scalar>
StepperOperatorSplitObserverComposite<Scalar>::
StepperOperatorSplitObserverComposite(){}

template<class Scalar>
StepperOperatorSplitObserverComposite<Scalar>::
~StepperOperatorSplitObserverComposite(){}

template<class Scalar>
void StepperOperatorSplitObserverComposite<Scalar>::
observeBeginTakeStep(
    Teuchos::RCP<SolutionHistory<Scalar> >  sh ,
    Stepper<Scalar> &  stepper )
{
  for(auto& o: observers_)
    o->observeBeginTakeStep(sh, stepper);
}

template<class Scalar>
void StepperOperatorSplitObserverComposite<Scalar>::
observeBeforeStepper(int  index ,
    Teuchos::RCP<SolutionHistory<Scalar> > sh ,
    StepperOperatorSplit<Scalar> & stepperOS )
{
  for(auto& o: observers_)
    o->observeBeforeStepper(index, sh, stepperOS);
}

template<class Scalar>
void StepperOperatorSplitObserverComposite<Scalar>::
observeAfterStepper(int  index ,
    Teuchos::RCP<SolutionHistory<Scalar> > sh ,
    StepperOperatorSplit<Scalar> & stepperOS)
{
  for(auto& o: observers_)
    o->observeAfterStepper(index, sh, stepperOS);
}

template<class Scalar>
void StepperOperatorSplitObserverComposite<Scalar>::
observeEndTakeStep(
    Teuchos::RCP<SolutionHistory<Scalar> >  sh ,
    Stepper<Scalar> & stepperOS )
{
  for(auto& o: observers_)
    o->observeEndTakeStep( sh, stepperOS);
}

//private:

  //std::vector<Teuchos::RCP<StepperOperatorSplitObserver<Scalar> > > observers_;

};
} // namespace Tempus
#endif // Tempus_StepperOperatorSplitObserverComposite_impl_hpp
