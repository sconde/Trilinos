#include "ShyLU_NodeTacho_config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>

#include "TachoExp_Util.hpp"
#include "TachoExp_DenseMatrixView.hpp"
#include "TachoExp_DenseFlopCount.hpp"

#include "TachoExp_Chol_ByBlocks.hpp"
#include "TachoExp_Gemm_ByBlocks.hpp"
#include "TachoExp_Herk_ByBlocks.hpp"
#include "TachoExp_Trsm_ByBlocks.hpp"

#include "TachoExp_CommandLineParser.hpp" 

#ifdef TACHO_HAVE_MKL
#include "mkl_service.h"
#endif

using namespace Tacho;
using namespace Tacho::Experimental;

#define PRINT_TIMER                                                     \
  printf("  Time \n");                                                  \
  printf("       byblocks/reference (speedup):                   %10.6f\n", t_reference/t_byblocks); \
  printf("\n");                                                         

int main (int argc, char *argv[]) {
  CommandLineParser opts("This example program measure the performance of dense-by-blocks on Kokkos::OpenMP");  

  bool serial = false;
  int nthreads = 1;
  bool verbose = true;
  int mbeg = 1000;
  int mend = 6000;
  int step = 1000;
  int mb = 128;

  opts.set_option<bool>("serial", "Flag for invoking serial algorithm", &serial);
  opts.set_option<int>("kokkos-threads", "Number of threads", &nthreads);
  opts.set_option<bool>("verbose", "Flag for verbose printing", &verbose);
  opts.set_option<int>("begin", "Test problem begin size", &mbeg);  
  opts.set_option<int>("end", "Test problem end size", &mend);  
  opts.set_option<int>("step", "Test problem step size", &step);  
  opts.set_option<int>("mb", "Blocksize", &mb);  

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return 

  Kokkos::initialize(argc, argv);

  typedef double value_type;
  typedef Kokkos::pair<ordinal_type,ordinal_type> range_type;
  typedef Kokkos::DefaultExecutionSpace exec_space;
  typedef Kokkos::DefaultHostExecutionSpace host_exec_space;

  printExecSpaceConfiguration<host_exec_space>("Default HostSpace");
  printExecSpaceConfiguration<     exec_space>("Default DeviceSpace");
  
  int r_val = 0;
  
  {
    typedef DenseMatrixView<value_type,exec_space>                   DenseMatrixViewType;
    typedef DenseMatrixView<DenseMatrixViewType,exec_space>          DenseMatrixOfBlocksType;

    typedef DenseMatrixView<value_type,host_exec_space>              DenseMatrixViewHostType;
    typedef DenseMatrixView<DenseMatrixViewHostType,host_exec_space> DenseMatrixOfBlocksHostType;

    Kokkos::Impl::Timer timer;

    typedef Kokkos::TaskScheduler<exec_space> sched_type;
    sched_type sched;

    typedef TaskFunctor_Chol<sched_type,DenseMatrixOfBlocksType,
      Uplo::Upper,Algo::ByBlocks> task_functor_chol;
    typedef TaskFunctor_Trsm<sched_type,double,DenseMatrixOfBlocksType,
      Side::Left,Uplo::Upper,Trans::ConjTranspose,Diag::NonUnit,Algo::ByBlocks> task_functor_trsm;
    typedef TaskFunctor_Gemm<sched_type,double,DenseMatrixOfBlocksType,
      Trans::NoTranspose,Trans::NoTranspose,Algo::ByBlocks> task_functor_gemm;

    const ordinal_type max_functor_size = 4*sizeof(task_functor_gemm);
    
    Kokkos::DualView<value_type*,exec_space> 
      a("a", mend*mend), a1("a1", mend*mend), a2("a2", mend*mend), 
      b("b", mend*mend);
    
    const ordinal_type bmend = (mend/mb) + 1;
    Kokkos::DualView<DenseMatrixViewType*,exec_space> 
      ha("ha", bmend*bmend), hb("hb", bmend*bmend), hc("hc", bmend*bmend);

    {
      const ordinal_type
        task_queue_capacity = bmend*bmend*bmend*max_functor_size,
        min_block_size  = 16,
        max_block_size  = 4*max_functor_size,
        num_superblock  = 4,
        superblock_size = task_queue_capacity/num_superblock;
      
      sched = sched_type(typename exec_space::memory_space(),
                         task_queue_capacity,
                         min_block_size,
                         max_block_size,
                         superblock_size);
    }

    const ordinal_type dry = -2, niter = 3;

    double t_reference = 0, t_byblocks = 0;

    Random<value_type> random;
    auto randomize = [&](const DenseMatrixViewHostType &mat) {
      const ordinal_type m = mat.dimension_0(), n = mat.dimension_1();
      for (ordinal_type j=0;j<n;++j)
        for (ordinal_type i=0;i<m;++i)
          mat(i,j) = random.value();
    };

    ///
    /// Chol
    ///
    for (ordinal_type m=mbeg;m<=mend;m+=step) {
      t_reference = 0; t_byblocks = 0;
      auto sub_a  = Kokkos::subview(a,  range_type(0,m*m));
      auto sub_a1 = Kokkos::subview(a1, range_type(0,m*m));
      auto sub_a2 = Kokkos::subview(a2, range_type(0,m*m));
      
      // test matrix generation
      {
        sub_a.template modify<host_exec_space>();

        Kokkos::deep_copy(sub_a.h_view, 0);
        DenseMatrixViewHostType A;
        A.set_view(m, m);
        A.attach_buffer(1, m, a.h_view.data());
        for (ordinal_type i=0;i<m;++i) {
          A(i,i) = 4;
          const ordinal_type ip = i+1;
          if (ip < m) {
            A(ip,i ) = 1;
            A(i ,ip) = 1;
          }
        }
      }

      // reference 
      {
        int dummy = 0;

        sub_a. template sync  <host_exec_space>();
        sub_a1.template modify<host_exec_space>();
        
        DenseMatrixViewHostType A;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a1.h_view.data());

        for (ordinal_type iter=dry;iter<niter;++iter) {
          Kokkos::deep_copy(sub_a1.h_view, sub_a.h_view);
          timer.reset();
          Chol<Uplo::Upper,Algo::External>::invoke(dummy, dummy, A);
          t_reference += (iter >= 0)*timer.seconds();
        }
        t_reference /= niter;
      }
      
      // dense by blocks
      {
        sub_a. template sync  <exec_space>();
        sub_a2.template modify<exec_space>();

        DenseMatrixViewType A;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a2.d_view.data());

        const ordinal_type bm = (m/mb) + (m%mb>0);

        DenseMatrixOfBlocksHostType HA;
        HA.set_view(bm, bm);
        HA.attach_buffer(1, bm, ha.h_view.data());
        
        DenseMatrixOfBlocksHostType DA;
        DA.set_view(bm, bm);
        DA.attach_buffer(1, bm, ha.d_view.data());
        
        {          
          ha.template modify<host_exec_space>();
          
          setMatrixOfBlocks(HA, m, m, mb);
          attachBaseBuffer(HA, A.data(), A.stride_0(), A.stride_1());

          ha.template sync<exec_space>();
          
          for (ordinal_type iter=dry;iter<niter;++iter) {
            Kokkos::deep_copy(sub_a2.d_view, sub_a.d_view);

            timer.reset();
            Kokkos::host_spawn(Kokkos::TaskTeam(sched, Kokkos::TaskPriority::High),
                               task_functor_chol(sched, DA));
            Kokkos::wait(sched);
            t_byblocks += (iter >=0)*timer.seconds();
            
            clearFutureOfBlocks(DA);
          }
          t_byblocks /= niter;
        }
      }

      {
        a1.template sync<host_exec_space>();
        a2.template sync<host_exec_space>();

        double diff = 0.0, norm = 0.0;
        for (ordinal_type p=0;p<(m*m);++p) {
          norm += a1.h_view(p)*a1.h_view(p);
          diff += (a1.h_view(p) - a2.h_view(p))*(a1.h_view(p) - a2.h_view(p));
        }
        const double relerr = sqrt(diff/norm), eps = std::numeric_limits<double>::epsilon()*1000;

        if (sqrt(diff/norm) > eps) {
          printf("******* chol problem %d fails, reltaive error against reference is %10.4f\n", 
                 m, relerr);
          r_val = -1;
          break;
        }
      }
      
      {
        const double kilo = 1024, gflop = DenseFlopCount<value_type>::Chol(m)/kilo/kilo/kilo;
        printf("chol problem %10d, gflop %10.2f, gflop/s :: reference %10.2f, byblocks %10.2f\n", 
               m, gflop, gflop/t_reference, gflop/t_byblocks);
        PRINT_TIMER;
      }
    }
    printf("\n\n");

    ///
    /// Trsm
    ///
    
    for (ordinal_type m=mbeg;m<=mend;m+=step) {
      t_reference = 0; t_byblocks = 0;
      auto sub_a  = Kokkos::subview(a,  range_type(0,m*m));
      auto sub_a1 = Kokkos::subview(a1, range_type(0,m*m));
      auto sub_a2 = Kokkos::subview(a2, range_type(0,m*m));

      {
        sub_a. template modify<host_exec_space>();
        sub_a1.template modify<host_exec_space>();

        DenseMatrixViewHostType A, B;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.h_view.data());

        B.set_view(m, m);
        B.attach_buffer(1, m, sub_a1.h_view.data());

        randomize(A);
        randomize(B);

        sub_a2.template modify<exec_space>();
        Kokkos::deep_copy(sub_a2.d_view, sub_a1.h_view);
      }

      // reference 
      {
        int dummy = 0;
        
        sub_a. template sync  <host_exec_space>();
        sub_a1.template sync  <host_exec_space>();
        sub_a1.template modify<host_exec_space>();

        DenseMatrixViewHostType A, B;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.h_view.data());

        B.set_view(m, m);
        B.attach_buffer(1, m, sub_a1.h_view.data());
        
        const double alpha = -1.0;
        for (ordinal_type iter=dry;iter<niter;++iter) {
          timer.reset();
          Trsm<Side::Left,Uplo::Upper,Trans::ConjTranspose,Algo::External>
            ::invoke(dummy, dummy, Diag::NonUnit(), alpha, A, B);
          t_reference += (iter >= 0)*timer.seconds();
        }
        t_reference /= niter;
      }
      
      // dense by blocks
      {
        sub_a. template sync  <exec_space>();
        sub_a2.template sync  <exec_space>();
        sub_a2.template modify<exec_space>();

        DenseMatrixViewType A, B;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.d_view.data());

        B.set_view(m, m);
        B.attach_buffer(1, m, sub_a2.d_view.data());

        const ordinal_type bm = (m/mb) + (m%mb>0);

        ha.template modify<host_exec_space>();
        hb.template modify<host_exec_space>();

        DenseMatrixOfBlocksHostType HA, HB;

        HA.set_view(bm, bm);
        HA.attach_buffer(1, bm, ha.h_view.data());

        HB.set_view(bm, bm);
        HB.attach_buffer(1, bm, hb.h_view.data());

        setMatrixOfBlocks(HA, m, m, mb);
        attachBaseBuffer(HA, A.data(), A.stride_0(), A.stride_1());
        
        setMatrixOfBlocks(HB, m, m, mb);
        attachBaseBuffer(HB, B.data(), B.stride_0(), B.stride_1());
        
        ha.template sync<exec_space>();
        hb.template sync<exec_space>();

        DenseMatrixOfBlocksType DA, DB;
        
        DA.set_view(bm, bm);
        DA.attach_buffer(1, bm, ha.d_view.data());
        
        DB.set_view(bm, bm);
        DB.attach_buffer(1, bm, hb.d_view.data());

        {
          const double alpha = -1.0;
          for (ordinal_type iter=dry;iter<niter;++iter) {
            timer.reset();
            Kokkos::host_spawn(Kokkos::TaskTeam(sched, Kokkos::TaskPriority::High),
                               task_functor_trsm(sched, alpha, DA, DB));
            Kokkos::wait(sched);
            t_byblocks += (iter >=0)*timer.seconds();

            clearFutureOfBlocks(DB);
          }
          t_byblocks /= niter;
        }
      }
      
      {
        a1.template sync<host_exec_space>();
        a2.template sync<host_exec_space>();

        double diff = 0.0, norm = 0.0;
        for (ordinal_type p=0;p<(m*m);++p) {
          norm += a1.h_view(p)*a1.h_view(p);
          diff += (a1.h_view(p) - a2.h_view(p))*(a1.h_view(p) - a2.h_view(p));
        }
        const double relerr = sqrt(diff/norm), eps = std::numeric_limits<double>::epsilon()*1000;

        if (sqrt(diff/norm) > eps) 
          printf("******* trsm problem %d fails, reltaive error against reference is %10.4f\n", 
                 m, relerr);
      }
      
      {
        const double kilo = 1024, gflop = DenseFlopCount<value_type>::Trsm(true, m, m)/kilo/kilo/kilo;
        printf("trsm problem %10d, gflop %10.2f, gflop/s :: reference %10.2f, byblocks %10.2f\n", 
               m, gflop, gflop/t_reference, gflop/t_byblocks);
        PRINT_TIMER;
      }
    }
    printf("\n\n");

    ///
    /// Gemm
    ///
    
    for (ordinal_type m=mbeg;m<=mend;m+=step) {
      t_reference = 0; t_byblocks = 0;
      auto sub_a  = Kokkos::subview(a,  range_type(0,m*m));
      auto sub_b  = Kokkos::subview(b,  range_type(0,m*m));
      auto sub_a1 = Kokkos::subview(a1, range_type(0,m*m));
      auto sub_a2 = Kokkos::subview(a2, range_type(0,m*m));

      {
        sub_a. template modify<host_exec_space>();
        sub_b. template modify<host_exec_space>();
        sub_a1.template modify<host_exec_space>();
        
        DenseMatrixViewHostType A, B, C;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.h_view.data());

        B.set_view(m, m);
        B.attach_buffer(1, m, sub_b.h_view.data());

        C.set_view(m, m);
        C.attach_buffer(1, m, sub_a1.h_view.data());

        randomize(A);
        randomize(B);
        randomize(C);

        sub_a2.template modify<exec_space>();
        Kokkos::deep_copy(sub_a2.d_view, sub_a1.h_view);
      }

      // reference 
      {
        int dummy = 0;
        
        sub_a. template sync  <host_exec_space>();
        sub_b. template sync  <host_exec_space>();
        sub_a1.template sync  <host_exec_space>();
        sub_a1.template modify<host_exec_space>();

        DenseMatrixViewType A, B, C;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.h_view.data());
        
        B.set_view(m, m);
        B.attach_buffer(1, m, sub_b.h_view.data());

        C.set_view(m, m);
        C.attach_buffer(1, m, sub_a1.h_view.data());
        
        const double alpha = -1.0, beta = 1.0;

        for (ordinal_type iter=dry;iter<niter;++iter) {
          timer.reset();
          Gemm<Trans::NoTranspose,Trans::NoTranspose,Algo::External>
            ::invoke(dummy, dummy, alpha, A, B, beta, C);
          t_reference += (iter >= 0)*timer.seconds();
        }
        t_reference /= niter;
      }
      
      // dense by blocks
      {
        sub_a. template sync  <exec_space>();
        sub_b. template sync  <exec_space>();
        sub_a2.template sync  <exec_space>();
        sub_a2.template modify<exec_space>();

        DenseMatrixViewType A, B, C;
        A.set_view(m, m);
        A.attach_buffer(1, m, sub_a.d_view.data());

        B.set_view(m, m);
        B.attach_buffer(1, m, sub_b.d_view.data());

        C.set_view(m, m);
        C.attach_buffer(1, m, sub_a2.d_view.data());

        const ordinal_type bm = (m/mb) + (m%mb>0);

        ha.template modify<host_exec_space>();
        hb.template modify<host_exec_space>();
        hc.template modify<host_exec_space>();

        DenseMatrixOfBlocksHostType HA, HB, HC;

        HA.set_view(bm, bm);
        HA.attach_buffer(1, bm, ha.h_view.data());

        HB.set_view(bm, bm);
        HB.attach_buffer(1, bm, hb.h_view.data());

        HC.set_view(bm, bm);
        HC.attach_buffer(1, bm, hc.h_view.data());

        setMatrixOfBlocks(HA, m, m, mb);
        attachBaseBuffer(HA, A.data(), A.stride_0(), A.stride_1());
        
        setMatrixOfBlocks(HB, m, m, mb);
        attachBaseBuffer(HB, B.data(), B.stride_0(), B.stride_1());
        
        setMatrixOfBlocks(HC, m, m, mb);
        attachBaseBuffer(HC, C.data(), C.stride_0(), C.stride_1());

        ha.template sync<exec_space>();
        hb.template sync<exec_space>();
        hc.template sync<exec_space>();
        
        DenseMatrixOfBlocksType DA, DB, DC;
        
        DA.set_view(bm, bm);
        DA.attach_buffer(1, bm, ha.d_view.data());

        DB.set_view(bm, bm);
        DB.attach_buffer(1, bm, hb.d_view.data());

        DC.set_view(bm, bm);
        DC.attach_buffer(1, bm, hc.d_view.data());

        {
          const double alpha = -1.0, beta = 1.0;
          for (ordinal_type iter=dry;iter<niter;++iter) {
            timer.reset();
            Kokkos::host_spawn(Kokkos::TaskTeam(sched, Kokkos::TaskPriority::High),
                               task_functor_gemm(sched, alpha, DA, DB, beta, DC));
            Kokkos::wait(sched);
            t_byblocks += (iter >=0)*timer.seconds();

            clearFutureOfBlocks(HC);
          }
          t_byblocks /= niter;
        }
      }
      
      {
        a1.template sync<host_exec_space>();
        a2.template sync<host_exec_space>();

        double diff = 0.0, norm = 0.0;
        for (ordinal_type p=0;p<(m*m);++p) {
          norm += a1.h_view(p)*a1.h_view(p);
          diff += (a1.h_view(p) - a2.h_view(p))*(a1.h_view(p) - a2.h_view(p));
        }
        const double relerr = sqrt(diff/norm), eps = std::numeric_limits<double>::epsilon()*1000;

        if (sqrt(diff/norm) > eps) 
          printf("******* gemm problem %d fails, reltaive error against reference is %10.4f\n", 
                 m, relerr);
      }
      
      {
        const double kilo = 1024, gflop = DenseFlopCount<value_type>::Gemm(m, m, m)/kilo/kilo/kilo;
        printf("gemm problem %10d, gflop %10.2f, gflop/s :: reference %10.2f, byblocks %10.2f\n", 
               m, gflop, gflop/t_reference, gflop/t_byblocks);
        PRINT_TIMER;
      }
    }
    printf("\n\n");
  }
  Kokkos::finalize();

  return r_val;
}
