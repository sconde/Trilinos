// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperRKButcherTableau_hpp
#define Tempus_StepperRKButcherTableau_hpp

// disable clang warnings
#ifdef __clang__
#pragma clang system_header
#endif

//#include "Tempus_String_Utilities.hpp"
#include "Tempus_StepperExplicitRK_new.hpp"
#include "Tempus_StepperDIRK_new.hpp"
#include "Tempus_RKButcherTableau.hpp"

//#include "Teuchos_Assert.hpp"
//#include "Teuchos_as.hpp"
//#include "Teuchos_Describable.hpp"
//#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"
//#include "Teuchos_VerboseObject.hpp"
//#include "Teuchos_VerboseObjectParameterListHelpers.hpp"
//#include "Teuchos_SerialDenseMatrix.hpp"
//#include "Teuchos_SerialDenseVector.hpp"
//#include "Thyra_MultiVectorStdOps.hpp"


namespace Tempus {


// ----------------------------------------------------------------------------
/** \brief General Explicit Runge-Kutta Butcher Tableau
 *
 *  The format of the Butcher Tableau parameter list is
    \verbatim
      <Parameter name="A" type="string" value="# # # ;
                                               # # # ;
                                               # # #">
      <Parameter name="b" type="string" value="# # #">
      <Parameter name="c" type="string" value="# # #">
    \endverbatim
 *  Note the number of stages is implicit in the number of entries.
 *  The number of stages must be consistent.
 *
 *  Default tableau is RK4 (order=4):
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cccc}  0  &  0  &     &     &    \\
 *                        1/2 & 1/2 &  0  &     &    \\
 *                        1/2 &  0  & 1/2 &  0  &    \\
 *                         1  &  0  &  0  &  1  &  0 \\ \hline
 *                            & 1/6 & 1/3 & 1/3 & 1/6 \end{array}
 *  \f]
 */
template<class Scalar>
class StepperERK_General :
  virtual public StepperExplicitRK_new<Scalar>
{
public:
  StepperERK_General()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_General(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded,
    const Teuchos::SerialDenseMatrix<int,Scalar>& A,
    const Teuchos::SerialDenseVector<int,Scalar>& b,
    const Teuchos::SerialDenseVector<int,Scalar>& c,
    const int order,
    const int orderMin,
    const int orderMax,
    const Teuchos::SerialDenseVector<int,Scalar>& bstar)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);

    this->setTableau(A,b,c,order,orderMin,orderMax,bstar);

    TEUCHOS_TEST_FOR_EXCEPTION(
      this->tableau_->isImplicit() == true, std::logic_error,
      "Error - General ERK received an implicit Butcher Tableau!\n");
  }

  virtual std::string description() const { return "General ERK"; }

  virtual std::string getDescription() const
  {
    std::stringstream Description;
    Description << this->description() << "\n"
      << "The format of the Butcher Tableau parameter list is\n"
      << "  <Parameter name=\"A\" type=\"string\" value=\"# # # ;\n"
      << "                                           # # # ;\n"
      << "                                           # # #\"/>\n"
      << "  <Parameter name=\"b\" type=\"string\" value=\"# # #\"/>\n"
      << "  <Parameter name=\"c\" type=\"string\" value=\"# # #\"/>\n\n"
      << "Note the number of stages is implicit in the number of entries.\n"
      << "The number of stages must be consistent.\n"
      << "\n"
      << "Default tableau is RK4 (order=4):\n"
      << "c = [  0  1/2 1/2  1  ]'\n"
      << "A = [  0              ]\n"
      << "    [ 1/2  0          ]\n"
      << "    [  0  1/2  0      ]\n"
      << "    [  0   0   1   0  ]\n"
      << "b = [ 1/6 1/3 1/3 1/6 ]'";
    return Description.str();
  }

  void setupTableau()
  {
    if (this->tableau_ == Teuchos::null) {
      // Set tableau to the default if null, otherwise keep current tableau.
      auto t = rcp(new Explicit4Stage4thOrder_RKBT<Scalar>());
      this->tableau_ = rcp(new RKButcherTableau<Scalar>(
                                 t->A(),t->b(),t->c(),
                                 t->order(),t->orderMin(),t->orderMax(),
                                 t->bstar()));
    }
  }

  void setTableau(const Teuchos::SerialDenseMatrix<int,Scalar>& A,
                  const Teuchos::SerialDenseVector<int,Scalar>& b,
                  const Teuchos::SerialDenseVector<int,Scalar>& c,
                  const int order,
                  const int orderMin,
                  const int orderMax,
                  const Teuchos::SerialDenseVector<int,Scalar>&
                    bstar = Teuchos::SerialDenseVector<int,Scalar>())
  {
    this->tableau_ = rcp(new RKButcherTableau<Scalar>(
                               A,b,c,order,orderMin,orderMax,bstar));
  }

  virtual std::string getDefaultICConsistency() const { return "Consistent"; }

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
    this->getValidParametersBasicERK(pl);
    pl->set<std::string>("Initial Condition Consistency",
                         this->getDefaultICConsistency());

    // Tableau ParameterList
    Teuchos::RCP<Teuchos::ParameterList> tableauPL = Teuchos::parameterList();
    tableauPL->set<std::string>("A",
     "0.0 0.0 0.0 0.0; 0.5 0.0 0.0 0.0; 0.0 0.5 0.0 0.0; 0.0 0.0 1.0 0.0");
    tableauPL->set<std::string>("b",
     "0.166666666666667 0.333333333333333 0.333333333333333 0.166666666666667");
    tableauPL->set<std::string>("c", "0.0 0.5 0.5 1.0");
    tableauPL->set<int>("order", 4);
    tableauPL->set<std::string>("bstar", "");
    pl->set("Tableau", *tableauPL);

    return pl;
  }
};


// ----------------------------------------------------------------------------
/** \brief Backward Euler Runge-Kutta Butcher Tableau
 *
 *  The tableau for Backward Euler (order=1) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|c} 1 & 1 \\ \hline
 *                       & 1 \end{array}
 *  \f]
 */
template<class Scalar>
class StepperDIRK_BackwardEuler :
  virtual public StepperDIRK_new<Scalar>
{
  public:
  StepperDIRK_BackwardEuler()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperDIRK_BackwardEuler(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    const Teuchos::RCP<Thyra::NonlinearSolverBase<Scalar> >& solver,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded,
    bool zeroInitialGuess)
  {
    this->setup(appModel, obs, solver, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded, zeroInitialGuess);
    this->setupTableau();
  }

  std::string description() const { return "RK Backward Euler"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "c = [ 1 ]'\n"
                << "A = [ 1 ]\n"
                << "b = [ 1 ]'";
    return Description.str();
  }

  virtual bool getICConsistencyCheckDefault() const { return false; }

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
    this->getValidParametersBasicDIRK(pl);
    pl->set<bool>("Initial Condition Consistency Check",
                  this->getICConsistencyCheckDefault());
    return pl;
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    int NumStages = 1;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) = ST::one();

    // Fill b:
    b(0) = ST::one();

    // Fill c:
    c(0) = ST::one();

    int order = 1;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief Forward Euler Runge-Kutta Butcher Tableau
 *
 *  The tableau for Forward Euler (order=1) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|c} 0 & 0 \\ \hline
 *                       & 1 \end{array}
 *  \f]
 */
template<class Scalar>
class StepperERK_ForwardEuler :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_ForwardEuler()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_ForwardEuler(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const { return "RK Forward Euler"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "c = [ 0 ]'\n"
                << "A = [ 0 ]\n"
                << "b = [ 1 ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    Teuchos::SerialDenseMatrix<int,Scalar> A(1,1);
    Teuchos::SerialDenseVector<int,Scalar> b(1);
    Teuchos::SerialDenseVector<int,Scalar> c(1);
    A(0,0) = ST::zero();
    b(0) = ST::one();
    c(0) = ST::zero();
    int order = 1;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief Runge-Kutta 4th order Butcher Tableau
 *
 *  The tableau for RK4 (order=4) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cccc}  0  &  0  &     &     &    \\
 *                        1/2 & 1/2 &  0  &     &    \\
 *                        1/2 &  0  & 1/2 &  0  &    \\
 *                         1  &  0  &  0  &  1  &  0 \\ \hline
 *                            & 1/6 & 1/3 & 1/3 & 1/6 \end{array}
 *  \f]
 */
template<class Scalar>
class StepperERK_4Stage4thOrder :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_4Stage4thOrder()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_4Stage4thOrder(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const { return "RK Explicit 4 Stage"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "\"The\" Runge-Kutta Method (explicit):\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 1.2, pg 138\n"
                << "c = [  0  1/2 1/2  1  ]'\n"
                << "A = [  0              ] \n"
                << "    [ 1/2  0          ]\n"
                << "    [  0  1/2  0      ]\n"
                << "    [  0   0   1   0  ]\n"
                << "b = [ 1/6 1/3 1/3 1/6 ]'";
    return Description.str();
  }

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);
    const Scalar onesixth = one/(6*one);
    const Scalar onethird = one/(3*one);

    int NumStages = 4;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) =    zero; A(0,1) =    zero; A(0,2) = zero; A(0,3) = zero;
    A(1,0) = onehalf; A(1,1) =    zero; A(1,2) = zero; A(1,3) = zero;
    A(2,0) =    zero; A(2,1) = onehalf; A(2,2) = zero; A(2,3) = zero;
    A(3,0) =    zero; A(3,1) =    zero; A(3,2) =  one; A(3,3) = zero;

    // Fill b:
    b(0) = onesixth; b(1) = onethird; b(2) = onethird; b(3) = onesixth;

    // fill c:
    c(0) = zero; c(1) = onehalf; c(2) = onehalf; c(3) = one;

    int order = 4;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief Explicit RK Bogacki-Shampine Butcher Tableau
 *
 *  The tableau (order=3(2)) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T \\ \hline
 *      & \hat{b}^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cccc}  0  & 0    &     &     & \\
 *                        1/2 & 1/2  & 0   &     & \\
 *                        3/4 & 0    & 3/4 & 0   & \\
 *                         1  & 2/9  & 1/3 & 4/9 & 0 \\ \hline
 *                            & 2/9  & 1/3 & 4/9 & 0 \\
 *                            & 7/24 & 1/4 & 1/3 & 1/8 \end{array}
 *  \f]
 *  Reference:  P. Bogacki and L.F. Shampine.
 *              A 3(2) pair of Runge–Kutta formulas.
 *              Applied Mathematics Letters, 2(4):321 – 325, 1989.
 */
template<class Scalar>
class StepperERK_BogackiShampine32 :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_BogackiShampine32()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_BogackiShampine32(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const {return "Bogacki-Shampine 3(2) Pair";}

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "P. Bogacki and L.F. Shampine.\n"
                << "A 3(2) pair of Runge–Kutta formulas.\n"
                << "Applied Mathematics Letters, 2(4):321 – 325, 1989.\n"
                << "c =     [ 0     1/2  3/4   1  ]'\n"
                << "A =     [ 0                   ]\n"
                << "        [ 1/2    0            ]\n"
                << "        [  0    3/4   0       ]\n"
                << "        [ 2/9   1/3  4/9   0  ]\n"
                << "b     = [ 2/9   1/3  4/9   0  ]'\n"
                << "bstar = [ 7/24  1/4  1/3  1/8 ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    using Teuchos::as;
    int NumStages = 4;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> bstar(NumStages);

    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);
    const Scalar onethird = one/(3*one);
    const Scalar threefourths = (3*one)/(4*one);
    const Scalar twoninths = (2*one)/(9*one);
    const Scalar fourninths = (4*one)/(9*one);

    // Fill A:
    A(0,0) =     zero; A(0,1) =        zero; A(0,2) =      zero; A(0,3) = zero;
    A(1,0) =  onehalf; A(1,1) =        zero; A(1,2) =      zero; A(1,3) = zero;
    A(2,0) =     zero; A(2,1) =threefourths; A(2,2) =      zero; A(2,3) = zero;
    A(3,0) =twoninths; A(3,1) =    onethird; A(3,2) =fourninths; A(3,3) = zero;

    // Fill b:
    b(0) = A(3,0); b(1) = A(3,1); b(2) = A(3,2); b(3) = A(3,3);

    // Fill c:
    c(0) = zero; c(1) = onehalf; c(2) = threefourths; c(3) = one;

    // Fill bstar
    bstar(0) = as<Scalar>(7*one/(24*one));
    bstar(1) = as<Scalar>(1*one/(4*one));
    bstar(2) = as<Scalar>(1*one/(3*one));
    bstar(3) = as<Scalar>(1*one/(8*one));
    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order,bstar));
  }
};


// ----------------------------------------------------------------------------
/** \brief Explicit RK Merson Butcher Tableau
 *
 *  The tableau (order=4(5)) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T \\ \hline
 *      & \hat{b}^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|ccccc}  0 & 0    &     &      &     & \\
 *                        1/3 & 1/3  & 0   &      &     & \\
 *                        1/3 & 1/6  & 1/6 & 0    &     & \\
 *                        1/2 & 1/8  & 0   & 3/8  &     & \\
 *                         1  & 1/2  & 0   & -3/2 & 2   & \\ \hline
 *                            & 1/6  & 0   & 0    & 2/3 & 1/6 \\
 *                            & 1/10 & 0   & 3/10 & 2/5 & 1/5 \end{array}
 *  \f]
 *  Reference:  E. Hairer, S.P. Norsett, G. Wanner,
 *              "Solving Ordinary Differential Equations I:
 *              Nonstiff Problems", 2nd Revised Edition,
 *              Table 4.1, pg 167.
 *
 */
template<class Scalar>
class StepperERK_Merson45 :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_Merson45()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_Merson45(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const { return "Merson 4(5) Pair"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 4.1, pg 167\n"
                << "c =     [  0    1/3  1/3  1/2   1  ]'\n"
                << "A =     [  0                       ]\n"
                << "        [ 1/3    0                 ]\n"
                << "        [ 1/6   1/6   0            ]\n"
                << "        [ 1/8    0   3/8   0       ]\n"
                << "        [ 1/2    0  -3/2   2    0  ]\n"
                << "b     = [ 1/6    0    0   2/3  1/6 ]'\n"
                << "bstar = [ 1/10   0  3/10  2/5  1/5 ]'";
    return Description.str();
  }


protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    using Teuchos::as;
    int NumStages = 5;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages, true);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages, true);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages, true);
    Teuchos::SerialDenseVector<int,Scalar> bstar(NumStages, true);

    const Scalar one = ST::one();
    const Scalar zero = ST::zero();

    // Fill A:
    A(1,0) = as<Scalar>(one/(3*one));;

    A(2,0) = as<Scalar>(one/(6*one));;
    A(2,1) = as<Scalar>(one/(6*one));;

    A(3,0) = as<Scalar>(one/(8*one));;
    A(3,2) = as<Scalar>(3*one/(8*one));;

    A(4,0) = as<Scalar>(one/(2*one));;
    A(4,2) = as<Scalar>(-3*one/(2*one));;
    A(4,3) = 2*one;

    // Fill b:
    b(0) = as<Scalar>(one/(6*one));
    b(3) = as<Scalar>(2*one/(3*one));
    b(4) = as<Scalar>(one/(6*one));

    // Fill c:
    c(0) = zero;
    c(1) = as<Scalar>(1*one/(3*one));
    c(2) = as<Scalar>(1*one/(3*one));
    c(3) = as<Scalar>(1*one/(2*one));
    c(4) = one;

    // Fill bstar
    bstar(0) = as<Scalar>(1*one/(10*one));
    bstar(2) = as<Scalar>(3*one/(10*one));
    bstar(3) = as<Scalar>(2*one/(5*one));
    bstar(4) = as<Scalar>(1*one/(5*one));
    int order = 4;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order,bstar));
  }
};


// ----------------------------------------------------------------------------
/** \brief Explicit RK 3/8th Rule Butcher Tableau
 *
 *  The tableau (order=4) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cccc}  0  &  0  &     &     &    \\
 *                        1/3 & 1/3 &  0  &     &    \\
 *                        2/3 &-1/3 &  1  &  0  &    \\
 *                         1  &  1  & -1  &  1  &  0 \\ \hline
 *                            & 1/8 & 3/8 & 3/8 & 1/8 \end{array}
 *  \f]
 *  Reference:  E. Hairer, S.P. Norsett, G. Wanner,
 *              "Solving Ordinary Differential Equations I:
 *              Nonstiff Problems", 2nd Revised Edition,
 *              Table 1.2, pg 138.
 */
template<class Scalar>
class StepperERK_3_8Rule :
  virtual public StepperExplicitRK_new<Scalar>
{
public:

  StepperERK_3_8Rule()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_3_8Rule(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const { return "RK Explicit 3/8 Rule"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 1.2, pg 138\n"
                << "c = [  0  1/3 2/3  1  ]'\n"
                << "A = [  0              ]\n"
                << "    [ 1/3  0          ]\n"
                << "    [-1/3  1   0      ]\n"
                << "    [  1  -1   1   0  ]\n"
                << "b = [ 1/8 3/8 3/8 1/8 ]'";
    return Description.str();
  }


protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    using Teuchos::as;
    int NumStages = 4;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onethird     = as<Scalar>(one/(3*one));
    const Scalar twothirds    = as<Scalar>(2*one/(3*one));
    const Scalar oneeighth    = as<Scalar>(one/(8*one));
    const Scalar threeeighths = as<Scalar>(3*one/(8*one));

    // Fill A:
    A(0,0) =      zero; A(0,1) = zero; A(0,2) = zero; A(0,3) = zero;
    A(1,0) =  onethird; A(1,1) = zero; A(1,2) = zero; A(1,3) = zero;
    A(2,0) = -onethird; A(2,1) =  one; A(2,2) = zero; A(2,3) = zero;
    A(3,0) =       one; A(3,1) = -one; A(3,2) =  one; A(3,3) = zero;

    // Fill b:
    b(0) =oneeighth; b(1) =threeeighths; b(2) =threeeighths; b(3) =oneeighth;

    // Fill c:
    c(0) = zero; c(1) = onethird; c(2) = twothirds; c(3) = one;

    int order = 4;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit 4 Stage 3rd order by Runge
 *
 *  The tableau (order=3) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cccc}  0  &  0  &     &     &    \\
 *                        1/2 & 1/2 &  0  &     &    \\
 *                         1  &  0  &  1  &  0  &    \\
 *                         1  &  0  &  0  &  1  &  0 \\ \hline
 *                            & 1/6 & 2/3 &  0  & 1/6 \end{array}
 *  \f]
 *  Reference:  E. Hairer, S.P. Norsett, G. Wanner,
 *              "Solving Ordinary Differential Equations I:
 *              Nonstiff Problems", 2nd Revised Edition,
 *              Table 1.1, pg 135.
 */
template<class Scalar>
class StepperERK_4Stage3rdOrderRunge :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_4Stage3rdOrderRunge()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_4Stage3rdOrderRunge(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const
    { return "RK Explicit 4 Stage 3rd order by Runge"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 1.1, pg 135\n"
                << "c = [  0  1/2  1   1  ]'\n"
                << "A = [  0              ]\n"
                << "    [ 1/2  0          ]\n"
                << "    [  0   1   0      ]\n"
                << "    [  0   0   1   0  ]\n"
                << "b = [ 1/6 2/3  0  1/6 ]'";
    return Description.str();
  }
protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    int NumStages = 4;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    const Scalar one = ST::one();
    const Scalar onehalf = one/(2*one);
    const Scalar onesixth = one/(6*one);
    const Scalar twothirds = 2*one/(3*one);
    const Scalar zero = ST::zero();

    // Fill A:
    A(0,0) =    zero; A(0,1) = zero; A(0,2) = zero; A(0,3) = zero;
    A(1,0) = onehalf; A(1,1) = zero; A(1,2) = zero; A(1,3) = zero;
    A(2,0) =    zero; A(2,1) =  one; A(2,2) = zero; A(2,3) = zero;
    A(3,0) =    zero; A(3,1) = zero; A(3,2) =  one; A(3,3) = zero;

    // Fill b:
    b(0) = onesixth; b(1) = twothirds; b(2) = zero; b(3) = onesixth;

    // Fill c:
    c(0) = zero; c(1) = onehalf; c(2) = one; c(3) = one;

    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit 5 Stage 3rd order by Kinnmark and Gray
 *
 *  The tableau (order=3) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|ccccc}  0  &  0  &     &     &     &    \\
 *                         1/5 & 1/5 &  0  &     &     &    \\
 *                         1/5 &  0  & 1/5 &  0  &     &    \\
 *                         1/3 &  0  &  0  & 1/3 &  0  &    \\
 *                         2/3 &  0  &  0  &  0  & 2/3 &  0 \\ \hline
 *                             & 1/4 &  0  &  0  &  0  & 3/4 \end{array}
 *  \f]
 *  Reference:  Modified by P. Ullrich.  From the prim_advance_mod.F90
 *              routine in the HOMME atmosphere model code.
 */
template<class Scalar>
class StepperERK_5Stage3rdOrderKandG :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_5Stage3rdOrderKandG()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_5Stage3rdOrderKandG(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const
    { return "RK Explicit 5 Stage 3rd order by Kinnmark and Gray"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Kinnmark & Gray 5 stage, 3rd order scheme \n"
                << "Modified by P. Ullrich.  From the prim_advance_mod.F90 \n"
                << "routine in the HOMME atmosphere model code.\n"
                << "c = [  0  1/5  1/5  1/3  2/3  ]'\n"
                << "A = [  0                      ]\n"
                << "    [ 1/5  0                  ]\n"
                << "    [  0  1/5   0             ]\n"
                << "    [  0   0   1/3   0        ]\n"
                << "    [  0   0    0   2/3   0   ]\n"
                << "b = [ 1/4  0    0    0   3/4  ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    int NumStages = 5;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    const Scalar one = ST::one();
    const Scalar onefifth = one/(5*one);
    const Scalar onefourth = one/(4*one);
    const Scalar onethird = one/(3*one);
    const Scalar twothirds = 2*one/(3*one);
    const Scalar threefourths = 3*one/(4*one);
    const Scalar zero = ST::zero();

    // Fill A:
    A(0,0) =     zero; A(0,1) =     zero; A(0,2) =     zero; A(0,3) =      zero; A(0,4) = zero;
    A(1,0) = onefifth; A(1,1) =     zero; A(1,2) =     zero; A(1,3) =      zero; A(1,4) = zero;
    A(2,0) =     zero; A(2,1) = onefifth; A(2,2) =     zero; A(2,3) =      zero; A(2,4) = zero;
    A(3,0) =     zero; A(3,1) =     zero; A(3,2) = onethird; A(3,3) =      zero; A(3,4) = zero;
    A(4,0) =     zero; A(4,1) =     zero; A(4,2) =     zero; A(4,3) = twothirds; A(4,4) = zero;

    // Fill b:
    b(0) =onefourth; b(1) =zero; b(2) =zero; b(3) =zero; b(4) =threefourths;

    // Fill c:
    c(0) =zero; c(1) =onefifth; c(2) =onefifth; c(3) =onethird; c(4) =twothirds;

    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit 3 Stage 3rd order
 *
 *  The tableau (order=3) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|ccc}  0  &  0  &     &     \\
 *                       1/2 & 1/2 &  0  &     \\
 *                        1  & -1  &  2  &  0  \\ \hline
 *                           & 1/6 & 4/6 & 1/6  \end{array}
 *  \f]
 */
template<class Scalar>
class StepperERK_3Stage3rdOrder :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_3Stage3rdOrder()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_3Stage3rdOrder(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const
    { return "RK Explicit 3 Stage 3rd order"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "c = [  0  1/2  1  ]'\n"
                << "A = [  0          ]\n"
                << "    [ 1/2  0      ]\n"
                << "    [ -1   2   0  ]\n"
                << "b = [ 1/6 4/6 1/6 ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar two = Teuchos::as<Scalar>(2*one);
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);
    const Scalar onesixth = one/(6*one);
    const Scalar foursixth = 4*one/(6*one);

    int NumStages = 3;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) =    zero; A(0,1) = zero; A(0,2) = zero;
    A(1,0) = onehalf; A(1,1) = zero; A(1,2) = zero;
    A(2,0) =    -one; A(2,1) =  two; A(2,2) = zero;

    // Fill b:
    b(0) = onesixth; b(1) = foursixth; b(2) = onesixth;

    // fill c:
    c(0) = zero; c(1) = onehalf; c(2) = one;

    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit 3 Stage 3rd order TVD
 *
 *  The tableau (order=3) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|ccc}  0  &  0  &     &     \\
 *                        1  &  1  &  0  &     \\
 *                       1/2 & 1/4 & 1/4 &  0  \\ \hline
 *                           & 1/6 & 1/6 & 4/6  \end{array}
 *  \f]
 *  Reference: Sigal Gottlieb and Chi-Wang Shu,
 *             'Total Variation Diminishing Runge-Kutta Schemes',
 *             Mathematics of Computation,
 *             Volume 67, Number 221, January 1998, pp. 73-85.
 *
 *  This is also written in the following set of updates.
    \verbatim
      u1 = u^n + dt L(u^n)
      u2 = 3 u^n/4 + u1/4 + dt L(u1)/4
      u^(n+1) = u^n/3 + 2 u2/2 + 2 dt L(u2)/3
    \endverbatim
 */
template<class Scalar>
class StepperERK_3Stage3rdOrderTVD :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_3Stage3rdOrderTVD()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_3Stage3rdOrderTVD(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }


  virtual std::string description() const
    { return "RK Explicit 3 Stage 3rd order TVD"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                  << "Sigal Gottlieb and Chi-Wang Shu\n"
                  << "`Total Variation Diminishing Runge-Kutta Schemes'\n"
                  << "Mathematics of Computation\n"
                  << "Volume 67, Number 221, January 1998, pp. 73-85\n"
                  << "c = [  0   1  1/2 ]'\n"
                  << "A = [  0          ]\n"
                  << "    [  1   0      ]\n"
                  << "    [ 1/4 1/4  0  ]\n"
                  << "b = [ 1/6 1/6 4/6 ]'\n"
                  << "This is also written in the following set of updates.\n"
                  << "u1 = u^n + dt L(u^n)\n"
                  << "u2 = 3 u^n/4 + u1/4 + dt L(u1)/4\n"
                  << "u^(n+1) = u^n/3 + 2 u2/2 + 2 dt L(u2)/3";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);
    const Scalar onefourth = one/(4*one);
    const Scalar onesixth = one/(6*one);
    const Scalar foursixth = 4*one/(6*one);

    int NumStages = 3;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) =      zero; A(0,1) =      zero; A(0,2) = zero;
    A(1,0) =       one; A(1,1) =      zero; A(1,2) = zero;
    A(2,0) = onefourth; A(2,1) = onefourth; A(2,2) = zero;

    // Fill b:
    b(0) = onesixth; b(1) = onesixth; b(2) = foursixth;

    // fill c:
    c(0) = zero; c(1) = one; c(2) = onehalf;

    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit 3 Stage 3rd order by Heun
 *
 *  The tableau (order=3) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|ccc}  0  &  0  &     &     \\
 *                       1/3 & 1/3 &  0  &     \\
 *                       2/3 &  0  & 2/3 &  0  \\ \hline
 *                           & 1/4 &  0  & 3/4  \end{array}
 *  \f]
 *  Reference:  E. Hairer, S.P. Norsett, G. Wanner,
 *              "Solving Ordinary Differential Equations I:
 *              Nonstiff Problems", 2nd Revised Edition,
 *              Table 1.1, pg 135.
 */
template<class Scalar>
class StepperERK_3Stage3rdOrderHeun :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_3Stage3rdOrderHeun()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_3Stage3rdOrderHeun(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const
    { return "RK Explicit 3 Stage 3rd order by Heun"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 1.1, pg 135\n"
                << "c = [  0  1/3 2/3 ]'\n"
                << "A = [  0          ] \n"
                << "    [ 1/3  0      ]\n"
                << "    [  0  2/3  0  ]\n"
                << "b = [ 1/4  0  3/4 ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onethird = one/(3*one);
    const Scalar twothirds = 2*one/(3*one);
    const Scalar onefourth = one/(4*one);
    const Scalar threefourths = 3*one/(4*one);

    int NumStages = 3;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) =     zero; A(0,1) =      zero; A(0,2) = zero;
    A(1,0) = onethird; A(1,1) =      zero; A(1,2) = zero;
    A(2,0) =     zero; A(2,1) = twothirds; A(2,2) = zero;

    // Fill b:
    b(0) = onefourth; b(1) = zero; b(2) = threefourths;

    // fill c:
    c(0) = zero; c(1) = onethird; c(2) = twothirds;

    int order = 3;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit Midpoint
 *
 *  The tableau (order=2) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cc}  0  &  0  &     \\
 *                      1/2 & 1/2 &  0  \\ \hline
 *                          &  0  &  1   \end{array}
 *  \f]
 *  Reference:  E. Hairer, S.P. Norsett, G. Wanner,
 *              "Solving Ordinary Differential Equations I:
 *              Nonstiff Problems", 2nd Revised Edition,
 *              Table 1.1, pg 135.
 */
template<class Scalar>
class StepperERK_Midpoint :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_Midpoint()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_Midpoint(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const { return "RK Explicit Midpoint"; }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "Solving Ordinary Differential Equations I:\n"
                << "Nonstiff Problems, 2nd Revised Edition\n"
                << "E. Hairer, S.P. Norsett, G. Wanner\n"
                << "Table 1.1, pg 135\n"
                << "c = [  0  1/2 ]'\n"
                << "A = [  0      ]\n"
                << "    [ 1/2  0  ]\n"
                << "b = [  0   1  ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
    typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);

    int NumStages = 2;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) =    zero; A(0,1) = zero;
    A(1,0) = onehalf; A(1,1) = zero;

    // Fill b:
    b(0) = zero; b(1) = one;

    // fill c:
    c(0) = zero; c(1) = onehalf;

    int order = 2;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief RK Explicit Trapezoidal
 *
 *  The tableau (order=2) is
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cc}  0  &  0  &     \\
 *                       1  &  1  &  0  \\ \hline
 *                          & 1/2 & 1/2  \end{array}
 *  \f]
 */
template<class Scalar>
class StepperERK_Trapezoidal :
  virtual public StepperExplicitRK_new<Scalar>
{
  public:
  StepperERK_Trapezoidal()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperERK_Trapezoidal(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded);
    this->setupTableau();
  }

  virtual std::string description() const
  {
    std::string stepperType = this->getStepperType();
    if ( stepperType == "" ) stepperType = "RK Explicit Trapezoidal";

    TEUCHOS_TEST_FOR_EXCEPTION(
      !( stepperType == "RK Explicit Trapezoidal" or
         stepperType == "Heuns Method")
      ,std::logic_error,
      "  ParameterList 'Stepper Type' (='" + stepperType + "')\n"
      "  does not match any name for this Stepper:\n"
      "    'RK Explicit Trapezoidal'\n"
      "    'Heuns Method'");

    return stepperType;
  }

  std::string getDescription() const
  {
    std::ostringstream Description;
    Description << this->description() << "\n"
                << "This Stepper is known as 'RK Explicit Trapezoidal' or 'Heuns Method'.\n"
                << "c = [  0   1  ]'\n"
                << "A = [  0      ]\n"
                << "    [  1   0  ]\n"
                << "b = [ 1/2 1/2 ]'";
    return Description.str();
  }

protected:

  void setupTableau()
  {
   typedef Teuchos::ScalarTraits<Scalar> ST;
    const Scalar one = ST::one();
    const Scalar zero = ST::zero();
    const Scalar onehalf = one/(2*one);

    int NumStages = 2;
    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);

    // Fill A:
    A(0,0) = zero; A(0,1) = zero;
    A(1,0) =  one; A(1,1) = zero;

    // Fill b:
    b(0) = onehalf; b(1) = onehalf;

    // fill c:
    c(0) = zero; c(1) = one;

    int order = 2;

    this->tableau_ = rcp(new RKButcherTableau<Scalar>(A,b,c,order,order,order));
  }
};


// ----------------------------------------------------------------------------
/** \brief General Implicit Runge-Kutta Butcher Tableau
 *
 *  The format of the Butcher Tableau parameter list is
    \verbatim
      <Parameter name="A" type="string" value="# # # ;
                                               # # # ;
                                               # # #">
      <Parameter name="b" type="string" value="# # #">
      <Parameter name="c" type="string" value="# # #">
    \endverbatim
 *  Note the number of stages is implicit in the number of entries.
 *  The number of stages must be consistent.
 *
 *  Default tableau is "SDIRK 2 Stage 2nd order":
 *  \f[
 *  \begin{array}{c|c}
 *    c & A \\ \hline
 *      & b^T
 *  \end{array}
 *  \;\;\;\;\mbox{ where }\;\;\;\;
 *  \begin{array}{c|cc} \gamma  & \gamma &        \\
 *                         1    & 1-\gamma & \gamma \\ \hline
 *                              & 1-\gamma & \gamma  \end{array}
 *  \f]
 *  where \f$\gamma = (2\pm \sqrt{2})/2\f$.  This will produce an
 *  L-stable 2nd order method.
 *
 *  Reference: U. M. Ascher and L. R. Petzold,
 *             Computer Methods for ODEs and DAEs, p. 106.
 */
template<class Scalar>
class StepperDIRK_General :
  virtual public StepperDIRK_new<Scalar>
{
  public:
  StepperDIRK_General()
  {
    this->setupDefault();
    this->setupTableau();
  }

  StepperDIRK_General(
    const Teuchos::RCP<const Thyra::ModelEvaluator<Scalar> >& appModel,
    const Teuchos::RCP<StepperExplicitRKObserverComposite<Scalar> >& obs,
    const Teuchos::RCP<Thyra::NonlinearSolverBase<Scalar> >& solver,
    bool useFSAL,
    std::string ICConsistency,
    bool ICConsistencyCheck,
    bool useEmbedded,
    bool zeroInitialGuess,
    const Teuchos::SerialDenseMatrix<int,Scalar>& A,
    const Teuchos::SerialDenseVector<int,Scalar>& b,
    const Teuchos::SerialDenseVector<int,Scalar>& c,
    const int order,
    const int orderMin,
    const int orderMax,
    const Teuchos::SerialDenseVector<int,Scalar>& bstar)
  {
    this->setup(appModel, obs, useFSAL, ICConsistency,
                ICConsistencyCheck, useEmbedded, zeroInitialGuess);

    this->setTableau(A,b,c,order,orderMin,orderMax,bstar);

    TEUCHOS_TEST_FOR_EXCEPTION(
      this->tableau_->isImplicit() != true, std::logic_error,
      "Error - General DIRK did not receive a DIRK Butcher Tableau!\n");
  }

  virtual std::string description() const { return "General DIRK"; }

  std::string getDescription() const
  {
    std::stringstream Description;
    Description << this->description() << "\n"
      << "The format of the Butcher Tableau parameter list is\n"
      << "  <Parameter name=\"A\" type=\"string\" value=\"# # # ;\n"
      << "                                           # # # ;\n"
      << "                                           # # #\"/>\n"
      << "  <Parameter name=\"b\" type=\"string\" value=\"# # #\"/>\n"
      << "  <Parameter name=\"c\" type=\"string\" value=\"# # #\"/>\n\n"
      << "Note the number of stages is implicit in the number of entries.\n"
      << "The number of stages must be consistent.\n"
      << "\n"
      << "Default tableau is 'SDIRK 2 Stage 2nd order':\n"
      << "  Computer Methods for ODEs and DAEs\n"
      << "  U. M. Ascher and L. R. Petzold\n"
      << "  p. 106\n"
      << "  gamma = (2-sqrt(2))/2\n"
      << "  c = [  gamma   1     ]'\n"
      << "  A = [  gamma   0     ]\n"
      << "      [ 1-gamma  gamma ]\n"
      << "  b = [ 1-gamma  gamma ]'";
    return Description.str();
  }

  virtual bool getICConsistencyCheckDefault() const { return false; }

  void setupTableau()
  {
    if (this->tableau_ == Teuchos::null) {
      // Set tableau to the default if null, otherwise keep current tableau.
      auto t = rcp(new SDIRK2Stage2ndOrder_RKBT<Scalar>());
      this->tableau_ = rcp(new RKButcherTableau<Scalar>(
                                 t->A(),t->b(),t->c(),
                                 t->order(),t->orderMin(),t->orderMax(),
                                 t->bstar()));
    }
  }

  void setTableau(const Teuchos::SerialDenseMatrix<int,Scalar>& A,
                  const Teuchos::SerialDenseVector<int,Scalar>& b,
                  const Teuchos::SerialDenseVector<int,Scalar>& c,
                  const int order,
                  const int orderMin,
                  const int orderMax,
                  const Teuchos::SerialDenseVector<int,Scalar>&
                    bstar = Teuchos::SerialDenseVector<int,Scalar>())
  {
    this->tableau_ = rcp(new RKButcherTableau<Scalar>(
                               A,b,c,order,orderMin,orderMax,bstar));
  }

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
    this->getValidParametersBasicDIRK(pl);
    pl->set<bool>("Initial Condition Consistency Check",
                  this->getICConsistencyCheckDefault());

    // Tableau ParameterList
    Teuchos::RCP<Teuchos::ParameterList> tableauPL = Teuchos::parameterList();
    tableauPL->set<std::string>("A",
     "0.2928932188134524 0.0; 0.7071067811865476 0.2928932188134524");
    tableauPL->set<std::string>("b",
     "0.7071067811865476 0.2928932188134524");
    tableauPL->set<std::string>("c", "0.2928932188134524 1.0");
    tableauPL->set<int>("order", 2);
    tableauPL->set<std::string>("bstar", "");
    pl->set("Tableau", *tableauPL);

    return pl;
  }
};


//// ----------------------------------------------------------------------------
///** \brief SDIRK 2 Stage 2nd order
// *
// *  The tableau (order=1 or 2) is
// *  \f[
// *  \begin{array}{c|c}
// *    c & A \\ \hline
// *      & b^T
// *  \end{array}
// *  \;\;\;\;\mbox{ where }\;\;\;\;
// *  \begin{array}{c|cc} \gamma  & \gamma &        \\
// *                         1    & 1-\gamma & \gamma \\ \hline
// *                              & 1-\gamma & \gamma  \end{array}
// *  \f]
// *  The default value is \f$\gamma = (2\pm \sqrt{2})/2\f$.
// *  This will produce an L-stable 2nd order method with the stage
// *  times within the timestep.  Other values of gamma will still
// *  produce an L-stable scheme, but will only be 1st order accurate.
// *  L-stability is guaranteed because \f$A_{sj} = b_j\f$.
// *
// *  Reference: U. M. Ascher and L. R. Petzold,
// *             Computer Methods for ODEs and DAEs, p. 106.
// */
//template<class Scalar>
//class SDIRK2Stage2ndOrder_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  SDIRK2Stage2ndOrder_RKBT()
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    const Scalar one = ST::one();
//    gamma_default_ = Teuchos::as<Scalar>((2*one-ST::squareroot(2*one))/(2*one));
//
//    this->setParameterList(Teuchos::null);
//  }
//
//  SDIRK2Stage2ndOrder_RKBT(Scalar gamma)
//  {
//    setGamma(gamma);
//  }
//
//  void setGamma(Scalar gamma)
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//    gamma_default_ = Teuchos::as<Scalar>((2*one-ST::squareroot(2*one))/(2*one));
//
//    RKButcherTableau<Scalar>::setParameterList(Teuchos::null);
//    this->RK_stepperPL_->template set<double>("gamma", gamma);
//    gamma_ = gamma;
//
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//
//    // Fill A:
//    A(0,0) =                              gamma; A(0,1) = zero;
//    A(1,0) = Teuchos::as<Scalar>( one - gamma ); A(1,1) = gamma;
//
//    // Fill b:
//    b(0) = Teuchos::as<Scalar>( one - gamma ); b(1) = gamma;
//
//    // Fill c:
//    c(0) = gamma; c(1) = one;
//
//    int order = 1;
//    if ( std::abs((gamma-gamma_default_)/gamma) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A,b,c,order,1,2);
//  }
//
//  virtual std::string description() const { return "SDIRK 2 Stage 2nd order"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Computer Methods for ODEs and DAEs\n"
//                << "U. M. Ascher and L. R. Petzold\n"
//                << "p. 106\n"
//                << "gamma = (2+-sqrt(2))/2\n"
//                << "c = [  gamma   1     ]'\n"
//                << "A = [  gamma   0     ]\n"
//                << "    [ 1-gamma  gamma ]\n"
//                << "b = [ 1-gamma  gamma ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//    Scalar gamma = pl->get<double>("gamma", gamma_default_);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//
//    // Fill A:
//    A(0,0) =                              gamma; A(0,1) = zero;
//    A(1,0) = Teuchos::as<Scalar>( one - gamma ); A(1,1) = gamma;
//
//    // Fill b:
//    b(0) = Teuchos::as<Scalar>( one - gamma ); b(1) = gamma;
//
//    // Fill c:
//    c(0) = gamma; c(1) = one;
//
//    int order = 1;
//    if ( std::abs((gamma-gamma_default_)/gamma) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A,b,c,order,1,2);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    pl->set<bool>("Initial Condition Consistency Check", false);
//    pl->set<double>("gamma",gamma_default_,
//      "The default value is gamma = (2-sqrt(2))/2. "
//      "This will produce an L-stable 2nd order method with the stage "
//      "times within the timestep.  Other values of gamma will still "
//      "produce an L-stable scheme, but will only be 1st order accurate.");
//
//    return pl;
//  }
//
//  private:
//    Scalar gamma_default_;
//    Scalar gamma_;
//};
//
//
//// ----------------------------------------------------------------------------
///** \brief SDIRK 2 Stage 3rd order
// *
// *  The tableau (order=2 or 3) is
// *  \f[
// *  \begin{array}{c|c}
// *    c & A \\ \hline
// *      & b^T
// *  \end{array}
// *  \;\;\;\;\mbox{ where }\;\;\;\;
// *  \begin{array}{c|cc}  \gamma  &  \gamma   &        \\
// *                      1-\gamma & 1-2\gamma & \gamma \\ \hline
// *                               &   1/2     &   1/2   \end{array}
// *  \f]
// *  \f[
// *  \gamma = \left\{ \begin{array}{cc}
// *                     (2\pm \sqrt{2})/2 & \mbox{then 2nd order and L-stable} \\
// *                     (3\pm \sqrt{3})/6 & \mbox{then 3rd order and A-stable}
// *                   \end{array} \right.
// *  \f]
// *  The default value is \f$\gamma = (3\pm \sqrt{3})/6\f$.
// *
// *  Reference: E. Hairer, S. P. Norsett, and G. Wanner,
// *             Solving Ordinary Differential Equations I:
// *             Nonstiff Problems, 2nd Revised Edition,
// *             Table 7.2, pg 207.
// */
//template<class Scalar>
//class SDIRK2Stage3rdOrder_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  SDIRK2Stage3rdOrder_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  SDIRK2Stage3rdOrder_RKBT(std::string gammaType,
//                           Scalar gamma = 0.7886751345948128)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      !(gammaType == "3rd Order A-stable" or
//        gammaType == "2nd Order L-stable" or
//        gammaType == "gamma"), std::logic_error,
//      "gammaType needs to be '3rd Order A-stable', '2nd Order L-stable' or 'gamma'.");
//
//    RKButcherTableau<Scalar>::setParameterList(Teuchos::null);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//
//    pl->set<std::string>("Gamma Type", gammaType);
//    gamma_ = gamma;
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//
//    int order = 0;
//    if (gammaType == "3rd Order A-stable") {
//      order = 3;
//      gamma_ = as<Scalar>((3*one+ST::squareroot(3*one))/(6*one));
//    } else if (gammaType == "2nd Order L-stable") {
//      order = 2;
//      gamma_ = as<Scalar>( (2*one - ST::squareroot(2*one))/(2*one) );
//    } else if (gammaType == "gamma") {
//      order = 2;
//      gamma_ = pl->get<double>("gamma",
//        as<Scalar>((3*one+ST::squareroot(3*one))/(6*one)));
//    }
//
//    // Fill A:
//    A(0,0) =                     gamma_; A(0,1) = zero;
//    A(1,0) = as<Scalar>(one - 2*gamma_); A(1,1) = gamma_;
//
//    // Fill b:
//    b(0) = as<Scalar>( one/(2*one) ); b(1) = as<Scalar>( one/(2*one) );
//
//    // Fill c:
//    c(0) = gamma_; c(1) = as<Scalar>( one - gamma_ );
//
//    this->setAbc(A,b,c,order,2,3);
//  }
//
//  virtual std::string description() const { return "SDIRK 2 Stage 3rd order"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Solving Ordinary Differential Equations I:\n"
//                << "Nonstiff Problems, 2nd Revised Edition\n"
//                << "E. Hairer, S. P. Norsett, and G. Wanner\n"
//                << "Table 7.2, pg 207\n"
//                << "gamma = (3+sqrt(3))/6 -> 3rd order and A-stable\n"
//                << "gamma = (2-sqrt(2))/2 -> 2nd order and L-stable\n"
//                << "c = [  gamma     1-gamma  ]'\n"
//                << "A = [  gamma     0        ]\n"
//                << "    [ 1-2*gamma  gamma    ]\n"
//                << "b = [ 1/2        1/2      ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//
//    std::string gammaType =
//      pl->get<std::string>("Gamma Type", "3rd Order A-stable");
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      !(gammaType == "3rd Order A-stable" or
//        gammaType == "2nd Order L-stable" or
//        gammaType == "gamma"), std::logic_error,
//      "gammaType needs to be '3rd Order A-stable', '2nd Order L-stable' or 'gamma'.");
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//
//    int order = 0;
//    Scalar gammaValue = 0.0;
//    if (gammaType == "3rd Order A-stable") {
//      order = 3;
//      gammaValue = as<Scalar>((3*one+ST::squareroot(3*one))/(6*one));
//    } else if (gammaType == "2nd Order L-stable") {
//      order = 2;
//      gammaValue = as<Scalar>( (2*one - ST::squareroot(2*one))/(2*one) );
//    } else if (gammaType == "gamma") {
//      order = 2;
//      gammaValue = pl->get<double>("gamma",
//        as<Scalar>((3*one+ST::squareroot(3*one))/(6*one)));
//    }
//
//    // Fill A:
//    A(0,0) =                     gammaValue; A(0,1) = zero;
//    A(1,0) = as<Scalar>(one - 2*gammaValue); A(1,1) = gammaValue;
//
//    // Fill b:
//    b(0) = as<Scalar>( one/(2*one) ); b(1) = as<Scalar>( one/(2*one) );
//
//    // Fill c:
//    c(0) = gammaValue; c(1) = as<Scalar>( one - gammaValue );
//
//    this->setAbc(A,b,c,order,2,3);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    pl->set<bool>("Initial Condition Consistency Check", false);
//    pl->set<std::string>("Gamma Type", "3rd Order A-stable",
//      "Valid values are '3rd Order A-stable' ((3+sqrt(3))/6.) "
//      "and '2nd Order L-stable' ((2-sqrt(2))/2).  The default "
//      "value is '3rd Order A-stable'");
//
//    return pl;
//  }
//
//  private:
//    Scalar gamma_default_;
//    Scalar gamma_;
//};
//
//
//// ----------------------------------------------------------------------------
///** \brief EDIRK 2 Stage 3rd order
// *
// *  The tableau (order=3) is
// *  \f[
// *  \begin{array}{c|c}
// *    c & A \\ \hline
// *      & b^T
// *  \end{array}
// *  \;\;\;\;\mbox{ where }\;\;\;\;
// *  \begin{array}{c|cc}  0  &  0  &     \\
// *                      2/3 & 1/3 & 1/3 \\ \hline
// *                          & 1/4 & 3/4  \end{array}
// *  \f]
// *  Reference: E. Hairer, S. P. Norsett, and G. Wanner,
// *             Solving Ordinary Differential Equations I:
// *             Nonstiff Problems, 2nd Revised Edition,
// *             Table 7.1, pg 205.
// */
//template<class Scalar>
//class EDIRK2Stage3rdOrder_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  EDIRK2Stage3rdOrder_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const { return "EDIRK 2 Stage 3rd order"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Hammer & Hollingsworth method\n"
//                << "Solving Ordinary Differential Equations I:\n"
//                << "Nonstiff Problems, 2nd Revised Edition\n"
//                << "E. Hairer, S. P. Norsett, and G. Wanner\n"
//                << "Table 7.1, pg 205\n"
//                << "c = [  0   2/3 ]'\n"
//                << "A = [  0    0  ]\n"
//                << "    [ 1/3  1/3 ]\n"
//                << "b = [ 1/4  3/4 ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//
//    // Fill A:
//    A(0,0) =                      zero; A(0,1) =                      zero;
//    A(1,0) = as<Scalar>( one/(3*one) ); A(1,1) = as<Scalar>( one/(3*one) );
//
//    // Fill b:
//    b(0) = as<Scalar>( one/(4*one) ); b(1) = as<Scalar>( 3*one/(4*one) );
//
//    // Fill c:
//    c(0) = zero; c(1) = as<Scalar>( 2*one/(3*one) );
//    int order = 3;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class IRK1StageTheta_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  IRK1StageTheta_RKBT()
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    theta_default_ = ST::one()/(2*ST::one());
//
//    this->setParameterList(Teuchos::null);
//  }
//
//  IRK1StageTheta_RKBT(Scalar theta)
//  {
//    setTheta(theta);
//  }
//
//  void setTheta(Scalar theta)
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    theta_default_ = ST::one()/(2*ST::one());
//
//    RKButcherTableau<Scalar>::setParameterList(Teuchos::null);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//    pl->set<double>("theta",theta);
//    theta_ = theta;
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 1;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    A(0,0) = theta;
//    b(0) = ST::one();
//    c(0) = theta;
//
//    int order = 1;
//    if ( std::abs((theta-theta_default_)/theta) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A, b, c, order, 1, 2);
//  }
//
//  virtual std::string description() const {return "IRK 1 Stage Theta Method";}
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Non-standard finite-difference methods\n"
//                << "in dynamical systems, P. Kama,\n"
//                << "Dissertation, University of Pretoria, pg. 49.\n"
//                << "Comment:  Generalized Implicit Midpoint Method\n"
//                << "c = [ theta ]'\n"
//                << "A = [ theta ]\n"
//                << "b = [  1  ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//    Scalar theta = pl->get<double>("theta",theta_default_);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 1;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    A(0,0) = theta;
//    b(0) = ST::one();
//    c(0) = theta;
//
//    int order = 1;
//    if ( std::abs((theta-theta_default_)/theta) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A, b, c, order, 1, 2);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    pl->set<bool>("Initial Condition Consistency Check", false);
//    pl->set<double>("theta",theta_default_,
//      "Valid values are 0 <= theta <= 1, where theta = 0 "
//      "implies Forward Euler, theta = 1/2 implies implicit midpoint "
//      "method (default), and theta = 1 implies Backward Euler. "
//      "For theta != 1/2, this method is first-order accurate, "
//      "and with theta = 1/2, it is second-order accurate.  "
//      "This method is A-stable, but becomes L-stable with theta=1.");
//
//    return pl;
//  }
//
//  private:
//    Scalar theta_default_;
//    Scalar theta_;
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class EDIRK2StageTheta_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  EDIRK2StageTheta_RKBT()
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    theta_default_ = ST::one()/(2*ST::one());
//
//    this->setParameterList(Teuchos::null);
//  }
//
//  EDIRK2StageTheta_RKBT(Scalar theta)
//  {
//    setTheta(theta);
//  }
//
//  void setTheta(Scalar theta)
//  {
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    theta_default_ = ST::one()/(2*ST::one());
//
//    RKButcherTableau<Scalar>::setParameterList(Teuchos::null);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//    pl->set<double>("theta", theta);
//    theta_ = theta;
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      theta == zero, std::logic_error,
//      "'theta' can not be zero, as it makes this IRK stepper explicit.");
//
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//
//    // Fill A:
//    A(0,0) =                               zero; A(0,1) =  zero;
//    A(1,0) = Teuchos::as<Scalar>( one - theta ); A(1,1) = theta;
//
//    // Fill b:
//    b(0) = Teuchos::as<Scalar>( one - theta );
//    b(1) = theta;
//
//    // Fill c:
//    c(0) = zero;
//    c(1) = one;
//
//    int order = 1;
//    if ( std::abs((theta-theta_default_)/theta) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A, b, c, order, 1, 2);
//  }
//
//  virtual std::string description() const {return "EDIRK 2 Stage Theta Method";}
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Computer Methods for ODEs and DAEs\n"
//                << "U. M. Ascher and L. R. Petzold\n"
//                << "p. 113\n"
//                << "c = [  0       1     ]'\n"
//                << "A = [  0       0     ]\n"
//                << "    [ 1-theta  theta ]\n"
//                << "b = [ 1-theta  theta ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//    Teuchos::RCP<Teuchos::ParameterList> pl = this->RK_stepperPL_;
//    Scalar theta = pl->get<double>("theta", theta_default_);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      theta == zero, std::logic_error,
//      "'theta' can not be zero, as it makes this IRK stepper explicit.");
//
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//
//    // Fill A:
//    A(0,0) =                               zero; A(0,1) =  zero;
//    A(1,0) = Teuchos::as<Scalar>( one - theta ); A(1,1) = theta;
//
//    // Fill b:
//    b(0) = Teuchos::as<Scalar>( one - theta );
//    b(1) = theta;
//
//    // Fill c:
//    c(0) = zero;
//    c(1) = one;
//
//    int order = 1;
//    if ( std::abs((theta-theta_default_)/theta) < 1.0e-08 ) order = 2;
//
//    this->setAbc(A, b, c, order, 1, 2);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    pl->set<bool>("Initial Condition Consistency Check", false);
//    pl->set<double>("theta",theta_default_,
//      "Valid values are 0 < theta <= 1, where theta = 0 "
//      "implies Forward Euler, theta = 1/2 implies trapezoidal "
//      "method (default), and theta = 1 implies Backward Euler. "
//      "For theta != 1/2, this method is first-order accurate, "
//      "and with theta = 1/2, it is second-order accurate.  "
//      "This method is A-stable, but becomes L-stable with theta=1.");
//
//    return pl;
//  }
//
//  private:
//    Scalar theta_default_;
//    Scalar theta_;
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class TrapezoidalRule_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  TrapezoidalRule_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const
//  {
//    std::string stepperType = "RK Trapezoidal Rule";
//    Teuchos::RCP<const Teuchos::ParameterList> pl = this->getParameterList();
//    if (pl != Teuchos::null) {
//      if (pl->isParameter("Stepper Type"))
//        stepperType = pl->get<std::string>("Stepper Type");
//    }
//
//    TEUCHOS_TEST_FOR_EXCEPTION(
//      !( stepperType == "RK Trapezoidal Rule" or
//         stepperType == "RK Crank-Nicolson")
//      ,std::logic_error,
//      "  ParameterList 'Stepper Type' (='" + stepperType + "')\n"
//      "  does not match any name for this Stepper:\n"
//      "    'RK Trapezoidal Rule'\n"
//      "    'RK Crank-Nicolson'");
//
//    return stepperType;
//  }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "Also known as Crank-Nicolson Method.\n"
//                << "c = [  0   1   ]'\n"
//                << "A = [  0   0   ]\n"
//                << "    [ 1/2  1/2 ]\n"
//                << "b = [ 1/2  1/2 ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//    const Scalar onehalf = ST::one()/(2*ST::one());
//
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//
//    // Fill A:
//    A(0,0) =    zero; A(0,1) =    zero;
//    A(1,0) = onehalf; A(1,1) = onehalf;
//
//    // Fill b:
//    b(0) = onehalf;
//    b(1) = onehalf;
//
//    // Fill c:
//    c(0) = zero;
//    c(1) = one;
//
//    int order = 2;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class ImplicitMidpoint_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  ImplicitMidpoint_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const { return "RK Implicit Midpoint"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "Table 5.2, pg 72\n"
//                << "Solving Ordinary Differential Equations I:\n"
//                << "Nonstiff Problems, 2nd Revised Edition\n"
//                << "E. Hairer, S. P. Norsett, and G. Wanner\n"
//                << "Table 7.1, pg 205\n"
//                << "c = [ 1/2 ]'\n"
//                << "A = [ 1/2 ]\n"
//                << "b = [  1  ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 1;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar onehalf = ST::one()/(2*ST::one());
//    const Scalar one = ST::one();
//
//    // Fill A:
//    A(0,0) = onehalf;
//
//    // Fill b:
//    b(0) = one;
//
//    // Fill c:
//    c(0) = onehalf;
//
//    int order = 2;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class Implicit1Stage1stOrderRadauA_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  Implicit1Stage1stOrderRadauA_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const
//    { return "RK Implicit 1 Stage 1st order Radau left"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "Table 5.3, pg 73\n"
//                << "c = [ 0 ]'\n"
//                << "A = [ 1 ]\n"
//                << "b = [ 1 ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 1;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//    A(0,0) = one;
//    b(0) = one;
//    c(0) = zero;
//    int order = 1;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class Implicit1Stage1stOrderRadauB_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  Implicit1Stage1stOrderRadauB_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const
//    { return "RK Implicit 1 Stage 1st order Radau right"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "Table 5.5, pg 74\n"
//                << "c = [ 1 ]'\n"
//                << "A = [ 1 ]\n"
//                << "b = [ 1 ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    int NumStages = 1;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar one = ST::one();
//    A(0,0) = one;
//    b(0) = one;
//    c(0) = one;
//    int order = 1;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class Implicit2Stage2ndOrderLobattoA_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  Implicit2Stage2ndOrderLobattoA_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const
//    { return "RK Implicit 2 Stage 2nd order Lobatto A"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "Table 5.7, pg 75\n"
//                << "c = [  0    1   ]'\n"
//                << "A = [  0    0   ]\n"
//                << "    [ 1/2  1/2  ]\n"
//                << "b = [ 1/2  1/2  ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar zero = ST::zero();
//    const Scalar one = ST::one();
//
//    // Fill A:
//    A(0,0) = zero;
//    A(0,1) = zero;
//    A(1,0) = as<Scalar>( one/(2*one) );
//    A(1,1) = as<Scalar>( one/(2*one) );
//
//    // Fill b:
//    b(0) = as<Scalar>( one/(2*one) );
//    b(1) = as<Scalar>( one/(2*one) );
//
//    // Fill c:
//    c(0) = zero;
//    c(1) = one;
//    int order = 2;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class Implicit2Stage2ndOrderLobattoB_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  Implicit2Stage2ndOrderLobattoB_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const
//    { return "RK Implicit 2 Stage 2nd order Lobatto B"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "Table 5.9, pg 76\n"
//                << "c = [  0    1   ]'\n"
//                << "A = [ 1/2   0   ]\n"
//                << "    [ 1/2   0   ]\n"
//                << "b = [ 1/2  1/2  ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar zero = ST::zero();
//    const Scalar one = ST::one();
//
//    // Fill A:
//    A(0,0) = as<Scalar>( one/(2*one) );
//    A(0,1) = zero;
//    A(1,0) = as<Scalar>( one/(2*one) );
//    A(1,1) = zero;
//
//    // Fill b:
//    b(0) = as<Scalar>( one/(2*one) );
//    b(1) = as<Scalar>( one/(2*one) );
//
//    // Fill c:
//    c(0) = zero;
//    c(1) = one;
//    int order = 2;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class SDIRK5Stage4thOrder_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  SDIRK5Stage4thOrder_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const { return "SDIRK 5 Stage 4th order"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//      << "L-stable\n"
//      << "Solving Ordinary Differential Equations II:\n"
//      << "Stiff and Differential-Algebraic Problems,\n"
//      << "2nd Revised Edition\n"
//      << "E. Hairer and G. Wanner\n"
//      << "pg100 \n"
//      << "c     = [ 1/4       3/4        11/20   1/2     1   ]'\n"
//      << "A     = [ 1/4                                      ]\n"
//      << "        [ 1/2       1/4                            ]\n"
//      << "        [ 17/50     -1/25      1/4                 ]\n"
//      << "        [ 371/1360  -137/2720  15/544  1/4         ]\n"
//      << "        [ 25/24     -49/48     125/16  -85/12  1/4 ]\n"
//      << "b     = [ 25/24     -49/48     125/16  -85/12  1/4 ]'\n"
//      << "bstar = [ 59/48     -17/96     225/32  -85/12  0   ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 5;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar zero = ST::zero();
//    const Scalar one = ST::one();
//    const Scalar onequarter = as<Scalar>( one/(4*one) );
//
//    // Fill A:
//    A(0,0) = onequarter;
//    A(0,1) = zero;
//    A(0,2) = zero;
//    A(0,3) = zero;
//    A(0,4) = zero;
//
//    A(1,0) = as<Scalar>( one / (2*one) );
//    A(1,1) = onequarter;
//    A(1,2) = zero;
//    A(1,3) = zero;
//    A(1,4) = zero;
//
//    A(2,0) = as<Scalar>( 17*one/(50*one) );
//    A(2,1) = as<Scalar>( -one/(25*one) );
//    A(2,2) = onequarter;
//    A(2,3) = zero;
//    A(2,4) = zero;
//
//    A(3,0) = as<Scalar>( 371*one/(1360*one) );
//    A(3,1) = as<Scalar>( -137*one/(2720*one) );
//    A(3,2) = as<Scalar>( 15*one/(544*one) );
//    A(3,3) = onequarter;
//    A(3,4) = zero;
//
//    A(4,0) = as<Scalar>( 25*one/(24*one) );
//    A(4,1) = as<Scalar>( -49*one/(48*one) );
//    A(4,2) = as<Scalar>( 125*one/(16*one) );
//    A(4,3) = as<Scalar>( -85*one/(12*one) );
//    A(4,4) = onequarter;
//
//    // Fill b:
//    b(0) = as<Scalar>( 25*one/(24*one) );
//    b(1) = as<Scalar>( -49*one/(48*one) );
//    b(2) = as<Scalar>( 125*one/(16*one) );
//    b(3) = as<Scalar>( -85*one/(12*one) );
//    b(4) = onequarter;
//
//    /*
//    // Alternate version
//    b(0) = as<Scalar>( 59*one/(48*one) );
//    b(1) = as<Scalar>( -17*one/(96*one) );
//    b(2) = as<Scalar>( 225*one/(32*one) );
//    b(3) = as<Scalar>( -85*one/(12*one) );
//    b(4) = zero;
//    */
//
//    // Fill c:
//    c(0) = onequarter;
//    c(1) = as<Scalar>( 3*one/(4*one) );
//    c(2) = as<Scalar>( 11*one/(20*one) );
//    c(3) = as<Scalar>( one/(2*one) );
//    c(4) = one;
//
//    int order = 4;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};
//
//
//// ----------------------------------------------------------------------------
//template<class Scalar>
//class SDIRK3Stage4thOrder_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  SDIRK3Stage4thOrder_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const { return "SDIRK 3 Stage 4th order"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "A-stable\n"
//                << "Solving Ordinary Differential Equations II:\n"
//                << "Stiff and Differential-Algebraic Problems,\n"
//                << "2nd Revised Edition\n"
//                << "E. Hairer and G. Wanner\n"
//                << "p. 100 \n"
//                << "gamma = (1/sqrt(3))*cos(pi/18)+1/2\n"
//                << "delta = 1/(6*(2*gamma-1)^2)\n"
//                << "c = [ gamma      1/2        1-gamma ]'\n"
//                << "A = [ gamma                         ]\n"
//                << "    [ 1/2-gamma  gamma              ]\n"
//                << "    [ 2*gamma    1-4*gamma  gamma   ]\n"
//                << "b = [ delta      1-2*delta  delta   ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 3;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    const Scalar zero = ST::zero();
//    const Scalar one = ST::one();
//    const Scalar pi = as<Scalar>(4*one)*std::atan(one);
//    const Scalar gamma = as<Scalar>( one/ST::squareroot(3*one)*std::cos(pi/(18*one))+one/(2*one) );
//    const Scalar delta = as<Scalar>( one/(6*one*std::pow(2*gamma-one,2*one)) );
//
//    // Fill A:
//    A(0,0) = gamma;
//    A(0,1) = zero;
//    A(0,2) = zero;
//
//    A(1,0) = as<Scalar>( one/(2*one) - gamma );
//    A(1,1) = gamma;
//    A(1,2) = zero;
//
//    A(2,0) = as<Scalar>( 2*gamma );
//    A(2,1) = as<Scalar>( one - 4*gamma );
//    A(2,2) = gamma;
//
//    // Fill b:
//    b(0) = delta;
//    b(1) = as<Scalar>( one-2*delta );
//    b(2) = delta;
//
//    // Fill c:
//    c(0) = gamma;
//    c(1) = as<Scalar>( one/(2*one) );
//    c(2) = as<Scalar>( one - gamma );
//
//    int order = 4;
//
//    this->setAbc(A,b,c,order);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};
//
//// ----------------------------------------------------------------------------
///** \brief SDIRK 2(1) pair
// *
// *  The tableau (order=2(1)) is
// *  \f[
// *  \begin{array}{c|c}
// *    c & A \\ \hline
// *      & b^T \\ \hline
// *      & \hat{b}^T
// *  \end{array}
// *  \;\;\;\;\mbox{ where }\;\;\;\;
// *  \begin{array}{c|cccc}  0 & 0   & \\
// *                         1 & -1  & 1 \\ \hline
// *                           & 1/2 & 1/2 \\
// *                           & 1   & 0 \end{array}
// *  \f]
// *
// */
//template<class Scalar>
//class SDIRK21_RKBT :
//  virtual public RKButcherTableau<Scalar>
//{
//  public:
//  SDIRK21_RKBT()
//  {
//    this->setParameterList(Teuchos::null);
//  }
//
//  virtual std::string description() const { return "SDIRK 2(1) Pair"; }
//
//  std::string getDescription() const
//  {
//    std::ostringstream Description;
//    Description << this->description() << "\n"
//                << "c =     [  1  0   ]'\n"
//                << "A =     [  1      ]\n"
//                << "        [ -1  1   ]\n"
//                << "b     = [ 1/2 1/2 ]'\n"
//                << "bstar = [  1  0   ]'";
//    return Description.str();
//  }
//
//  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& pList)
//  {
//    RKButcherTableau<Scalar>::setParameterList(pList);
//
//    typedef Teuchos::ScalarTraits<Scalar> ST;
//    using Teuchos::as;
//    int NumStages = 2;
//    Teuchos::SerialDenseMatrix<int,Scalar> A(NumStages,NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> b(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> c(NumStages);
//    Teuchos::SerialDenseVector<int,Scalar> bstar(NumStages);
//
//    const Scalar one = ST::one();
//    const Scalar zero = ST::zero();
//
//    // Fill A:
//    A(0,0) =  one; A(0,1) = zero;
//    A(1,0) = -one; A(1,1) =  one;
//
//    // Fill b:
//    b(0) = as<Scalar>(one/(2*one));
//    b(1) = as<Scalar>(one/(2*one));
//
//    // Fill c:
//    c(0) = one;
//    c(1) = zero;
//
//    // Fill bstar
//    bstar(0) = one;
//    bstar(1) = zero;
//    int order = 2;
//
//    this->setAbc(A,b,c,order,bstar);
//  }
//
//  Teuchos::RCP<const Teuchos::ParameterList>
//  getValidParameters() const
//  {
//    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
//    pl->setParameters( *(this->getValidParametersImplicit()));
//    pl->set<bool>("Initial Condition Consistency Check", false);
//
//    return pl;
//  }
//};


} // namespace Tempus


#endif // Tempus_StepperRKButcherTableau_hpp
