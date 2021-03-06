#ifndef __TACHOEXP_LAPACK_TEAM_HPP__
#define __TACHOEXP_LAPACK_TEAM_HPP__

/// \file  TachoExp_Lapack_TEAM.hpp
/// \brief BLAS wrapper
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "ShyLU_NodeTacho_config.h"
#include "TachoExp_Util.hpp"

namespace Tacho {

  namespace Experimental {
    
    template<typename T>
    struct LapackTeam {
      struct Impl {
        template<typename MemberType>
        static 
        KOKKOS_INLINE_FUNCTION
        void potrf_upper(const MemberType &member, 
                         const int m, 
                         T *A, const int as0, const int as1,
                         int *info) {
          if (m <= 0) return;
          typedef ArithTraits<T> ats;
          for (int p=0;p<m;++p) {
            const int jend = m-p-1;
            
            T
              *__restrict__ alpha11 = A+(p  )*as0+(p  )*as1,
              *__restrict__ a12t    = A+(p  )*as0+(p+1)*as1,
              *__restrict__ A22     = A+(p+1)*as0+(p+1)*as1;
            
            if (member.team_rank() == 0) 
              *alpha11 = sqrt(ats::real(*alpha11));
            member.team_barrier();

            const auto alpha = ats::real(*alpha11);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member,jend),[&](const int &j) {
                a12t[j*as1] /= alpha;
              });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member,jend),[&](const int &j) {
                const T aa = ats::conj(a12t[j*as1]);
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,j+1),[&](const int &i) {
                    const T bb = a12t[i*as1];
                    A22[i*as0+j*as1] -= aa*bb;
                  });
              });
          }
          
        }


        // template<typename MemberType>
        // static 
        // KOKKOS_INLINE_FUNCTION
        // int potrf_lower(const MemberType &member, 
        //                 const int m, 
        //                 const T *A, const int as0, const int as1) {
        //   if (m <= 0) return 0;
        //   typedef ArithTraits<T> ats;
        //   for (int p=0;p<m;++p) {
        //     const int jend = m-p-1;
            
        //     T &alpha11 = A[p*as0+p*as1];
        //     alpha11 = sqrt(real(alpha11));

        //     T
        //       *__restrict__ a21 = A+(p+1)*as0+(p  )*as1,
        //       *__restrict__ A22 = A+(p+1)*as0+(p+1)*as1;

        //     member.team_barrier();
        //     Kokkos::parallel_for(Kokkos::TeamThreadRange(member,p+1,m),[&](const int &i) {
        //         a21[i*as0] /= alpha11;
        //       });

        //     Kokkos::parallel_for(Kokkos::TeamThreadRange(member,jend),[&](const int &j) {
        //         Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,j+1),[&](const int &i) {
        //             const T
        //               *__restrict__ pA = A22+j*as0,
        //               *__restrict__ pB = A22+i*as0;
        //             T c(0);
        //             for (int p=0;p<k;++p)
        //               c += pA[p*as1]*pB[p*as1];
        //             A22[i*cs0+j*cs1] += c;
        //           });
        //       });
        //   }
        // }

      };

      template<typename MemberType>
      static 
      KOKKOS_INLINE_FUNCTION
      void potrf(const MemberType &member,
                 const char uplo,
                 const int m, 
                 /* */ T *A, const int lda,
                 int *info) {
        switch (uplo) {
        case 'U':
        case 'u': {
          Impl::potrf_upper(member, 
                            m,
                            A, 1, lda,
                            info); 
          break;
        }
        case 'L':
        case 'l': {
          Kokkos::abort("not implemented");
          break;
        }
        default:
          Kokkos::abort("Invalid uplo character");
        }
      }

    };
  }
}

#endif
