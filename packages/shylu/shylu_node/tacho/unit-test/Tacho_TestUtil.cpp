#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "TachoExp_Util.hpp"

using namespace Tacho::Experimental;

TEST( Util, is_complex_type ) {
  EXPECT_FALSE(int(ArithTraits<double>::is_complex));
  EXPECT_TRUE(int(ArithTraits<std::complex<double> >::is_complex));
  EXPECT_TRUE(int(ArithTraits<Kokkos::complex<double> >::is_complex));
}

TEST( Util, coo ) {
  {
    auto a = Coo<double>();
    EXPECT_EQ(a.i, 0);
    EXPECT_EQ(a.j, 0);
    EXPECT_EQ(a.j, 0.0);
  }
  {
    auto a = Coo<double>(1,3, 3.0);
    auto b = Coo<double>(1,3,10.0);
    auto c = Coo<double>(2,3, 3.0);
    auto d = Coo<double>(1,1, 3.0);
    
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a < c);
    EXPECT_FALSE(a < d);
  }
}

TEST( util, tag ) {
  // EXPECT_TRUE(is_valid_partition_tag<Partition::Top>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::Bottom>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::Left>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::Right>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::TopLeft>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::TopRight>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::BottomLeft>::value);
  // EXPECT_TRUE(is_valid_partition_tag<Partition::BottomRight>::value);

  EXPECT_TRUE(int(is_valid_uplo_tag<Uplo::Upper>::value));
  EXPECT_TRUE(int(is_valid_uplo_tag<Uplo::Lower>::value));

  EXPECT_TRUE(int(is_valid_side_tag<Side::Left>::value));
  EXPECT_TRUE(int(is_valid_side_tag<Side::Right>::value));

  EXPECT_TRUE(int(is_valid_diag_tag<Diag::Unit>::value));
  EXPECT_TRUE(int(is_valid_diag_tag<Diag::NonUnit>::value));

  EXPECT_TRUE(int(is_valid_trans_tag<Trans::Transpose>::value));
  EXPECT_TRUE(int(is_valid_trans_tag<Trans::ConjTranspose>::value));
  EXPECT_TRUE(int(is_valid_trans_tag<Trans::NoTranspose>::value));

  // EXPECT_FALSE(is_valid_partition_tag<NullTag>::value);
  EXPECT_FALSE(int(is_valid_uplo_tag<NullTag>::value));
  EXPECT_FALSE(int(is_valid_side_tag<NullTag>::value));
  EXPECT_FALSE(int(is_valid_diag_tag<NullTag>::value));
  EXPECT_FALSE(int(is_valid_trans_tag<NullTag>::value));
}

int main (int argc, char *argv[]) {
  
  typedef Kokkos::DefaultHostExecutionSpace HostSpaceType;
  typedef Kokkos::DefaultExecutionSpace     DeviceSpaceType;
  
  const bool detail = false;
  printExecSpaceConfiguration<DeviceSpaceType>("DeviceSpace", detail);
  printExecSpaceConfiguration<HostSpaceType>  ("HostSpace",   detail);
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
