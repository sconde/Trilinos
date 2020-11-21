// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_TimeStepControlStrategyComposite_hpp
#define Tempus_TimeStepControlStrategyComposite_hpp

#include "Tempus_TimeStepControlStrategy.hpp"
#include "Tempus_SolutionHistory.hpp"


namespace Tempus {

template<class Scalar> class TimeStepControl;

/** \brief TimeStepControlStrategyComposite loops over a vector of TimeStepControlStrategies.
 *
 *
 * Essentially, this is an <b>and</b> case if each strategies do a `min`
 * \f$ \Delta t = \min_{i \leq N} \{ \Delta t_i \}\f$
 *
 * The assumption is that each strategy will simply
 * update (or override) the step size `dt` with `metadata->setDt(dt)`
 * sequentially.
 *
 *  Examples of TimeStepControlStrategy:
 *   - TimeStepControlStrategyConstant
 *   - TimeStepControlStrategyBasicVS
 *   - TimeStepControlStrategyIntegralController
 *
 * <b>Note:</b> The ordering in the TimeStepControlStrategyComposite
 * list is very important.  The final TimeStepControlStrategy from
 * the composite could negate all previous step size updates.
 */
template<class Scalar>
class TimeStepControlStrategyComposite
  : virtual public TimeStepControlStrategy<Scalar>
{
public:

  /// Constructor
  TimeStepControlStrategyComposite(){}

  /// Destructor
  virtual ~TimeStepControlStrategyComposite(){}

  /** \brief Determine the time step size.*/
  virtual void setNextTimeStep(const TimeStepControl<Scalar> & tsc,
                               Teuchos::RCP<SolutionHistory<Scalar> > sh,
                               Status & integratorStatus) override
  {
    for(auto& s : strategies_)
      s->setNextTimeStep(tsc, sh, integratorStatus);
  }

  /// \name Overridden from Teuchos::Describable
  //@{
    std::string description() const override
    { return "Tempus::TimeStepControlComposite"; }

    void describe(Teuchos::FancyOStream          &out,
                  const Teuchos::EVerbosityLevel verbLevel) const override
    {
      Teuchos::OSTab ostab(out,2,"describe");
      out << description() << "::describe:" << std::endl;
      for(auto& s : strategies_)
        s->describe(out, verbLevel);
    }
  //@}

  /** \brief Append strategy to the composite list.*/
  void addStrategy(
    const Teuchos::RCP<TimeStepControlStrategy<Scalar> > &strategy)
  {
    if (Teuchos::nonnull(strategy))
      strategies_.push_back(strategy);
  }

  /** \brief Clear the composite list.*/
  void clearStrategies() { strategies_.clear(); }

  virtual void initialize() const override
  {
    for(auto& s : strategies_)
      s->initialize();

    if (strategies_.size() > 0) {
      auto strategy0 = strategies_[0];
      for (auto& s : strategies_) {
        if (strategy0->getStepType() != s->getStepType()) {
          std::ostringstream msg;
          msg << "Error - All the Strategy Step Types must match.\n";
          for(std::size_t i = 0; i < strategies_.size(); ++i) {
            msg << "  Strategy[" << i << "] = "
                << strategies_[i]->getStepType() << "\n";
          }
          TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error, msg.str());
        }
      }
    }

    this->isInitialized_ = true;   // Only place where this is set to true!
  }

private:

  std::vector<Teuchos::RCP<TimeStepControlStrategy<Scalar > > > strategies_;

};


} // namespace Tempus
#endif // Tempus_TimeStepControlStrategy_hpp
