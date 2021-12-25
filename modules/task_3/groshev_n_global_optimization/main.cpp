// Copyright 2021 Groshev Nikolay
#include <gtest/gtest.h>

#include <gtest-mpi-listener.hpp>

#include "../../../modules/task_3/groshev_n_global_optimization/opt.h"

double f1(double* x) { return *x; }

double f2(double* x) { return *x * *x; }

double f3(double* x) { return exp(*x); }

double f4(double* x) { return cos(*x); }

double f5(double* x) { return sin(*x); }

TEST(GlobalOptimization, f1) {
  int part = 1000;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double parall_res = perallel_global_opt(1, 3.7, f1, part, 0.001);
  if (ProcRank == 0) {
    double seq_res = seq_global_opt(1, 3.7, f1, part, 0.001);
    ASSERT_NEAR(parall_res, seq_res, 0.001);
  }
}

TEST(GlobalOptimization, f2) {
  int part = 1000;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double parall_res = perallel_global_opt(1, 500, f2, part, 0.001);
  if (ProcRank == 0) {
    double seq_res = seq_global_opt(1, 500, f2, part, 0.001);
    ASSERT_NEAR(parall_res, seq_res, 0.001);
  }
}

TEST(GlobalOptimization, f3) {
  int part = 1000;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double parall_res = perallel_global_opt(0.144, 5.77, f3, part, 0.001);
  if (ProcRank == 0) {
    double seq_res = seq_global_opt(0.144, 5.77, f3, part, 0.001);
    ASSERT_NEAR(parall_res, seq_res, 0.001);
  }
}

TEST(GlobalOptimization, f4) {
  int part = 1000;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double parall_res = perallel_global_opt(0.1, 0.11111, f4, part, 0.001);
  if (ProcRank == 0) {
    double seq_res = seq_global_opt(0.1, 0.11111, f4, part, 0.001);
    ASSERT_NEAR(parall_res, seq_res, 0.001);
  }
}

TEST(GlobalOptimization, f5) {
  int part = 1000;
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  double parall_res = perallel_global_opt(25, 27.679, f5, part, 0.001);
  if (ProcRank == 0) {
    double seq_res = seq_global_opt(25, 27.679, f5, part, 0.001);
    ASSERT_NEAR(parall_res, seq_res, 0.001);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
