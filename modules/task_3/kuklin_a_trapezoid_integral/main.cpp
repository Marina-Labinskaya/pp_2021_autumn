// Copyright 2021 Kuklin Andrey
#include <gtest/gtest.h>
#include <iostream>
#include "./trapezoid_integral.h"
#include <gtest-mpi-listener.hpp>

TEST(Integral_Sequential, can_solve_integral_sequential) {
  Func *f = new PiIntegral();
  ASSERT_NO_THROW(TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                              f->GetUpperBound(), 10));
}

TEST(Integral_Sequential, can_solve_integral_parallel) {
  Func *f = new PiIntegral();
  ASSERT_NO_THROW(
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 10));
}

TEST(Integral_Sequential, seq_parall_is_eqv) {
  Func *f = new PiIntegral();
  int procRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  double res_sequal = TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                                  f->GetUpperBound(), 100);
  double res_paral =
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 100);

  if (procRank == 0) {
    ASSERT_TRUE(fabs(res_paral - res_sequal) < 0.01);
  }
}

TEST(Integral_Sequential, first_integral) {
  Func *f = new FirstIntegral();
  int procRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  double res_sequal = TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                                  f->GetUpperBound(), 150);
  double res_paral =
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 150);

  if (procRank == 0) {
    ASSERT_TRUE(fabs(res_paral - res_sequal) < 0.01);
  }
}

TEST(Integral_Sequential, second_integral) {
  Func *f = new SecondIntegral();
  int procRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  double res_sequal = TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                                  f->GetUpperBound(), 200);
  double res_paral =
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 200);

  if (procRank == 0) {
    ASSERT_TRUE(fabs(res_paral - res_sequal) < 0.01);
  }
}

TEST(Integral_Sequential, third_integral) {
  Func *f = new ThirdIntegral();
  int procRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  double res_sequal = 0.0;
  if (procRank == 0) {
    res_sequal = TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                             f->GetUpperBound(), 100);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double res_paral =
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 100);

  if (procRank == 0) {
    ASSERT_TRUE(fabs(res_paral - res_sequal) < 0.01);
  }
}

TEST(Integral_Sequential, time_test) {
  Func *f = new ThirdIntegral();
  int procRank;
  double start_time, end_time, seq_time, par_time;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  double res_sequal = 0.0;
  if (procRank == 0) {
    start_time = MPI_Wtime();
    res_sequal = TrapezoidIntegralSequential(f, f->GetLowerBound(),
                                             f->GetUpperBound(), 100);
    end_time = MPI_Wtime();
    seq_time = end_time - start_time;
    std::cout << "time seq: " << seq_time << "\n";
    start_time = MPI_Wtime();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double res_paral =
      TrapezoidIntegralParallel(f, f->GetLowerBound(), f->GetUpperBound(), 100);

  if (procRank == 0) {
    end_time = MPI_Wtime();
    par_time = end_time - start_time;
    std::cout << "time par: " << par_time << "\n";
    std::cout << "parall value: " << seq_time / par_time << "\n";
    ASSERT_TRUE(fabs(res_paral - res_sequal) < 0.01);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners &listener =
      ::testing::UnitTest::GetInstance()->listeners();

  listener.Release(listener.default_result_printer());
  listener.Release(listener.default_xml_generator());

  listener.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
