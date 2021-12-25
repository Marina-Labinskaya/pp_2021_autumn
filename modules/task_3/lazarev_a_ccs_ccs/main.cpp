// Copyright 2021 Lazarev Alexey

#include <gtest/gtest.h>

#include <gtest-mpi-listener.hpp>

#include "../../../modules/task_3/lazarev_a_ccs_ccs/CCS.h"

TEST(Matrix_CCS, Error_size) {
  int ProcRank;
  matrix_CCS A, B;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    A = build_data_of_matrix_CCS(random_matrix(5, 5), 5, 5);
    B = build_data_of_matrix_CCS(random_matrix(4, 5), 4, 5);
    ASSERT_ANY_THROW(multiplication(A, B));
  }
}

TEST(Matrix_CCS, Error_size_2) {
  int ProcRank;
  matrix_CCS A, B;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    ASSERT_ANY_THROW(build_data_of_matrix_CCS(random_matrix(0, 0), 0, 0));
  }
}

TEST(Matrix_CCS, Matrix_1x1) {
  int ProcRank;
  matrix_CCS A, B;
  std::vector<double> parallel_mult;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    A = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    B = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    parallel_mult = parallel_multiplication(A, B);
  } else {
    A = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    B = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    parallel_mult = parallel_multiplication(A, B);
  }

  // parallel_mult = parallel_multiplication(A, B);

  if (ProcRank == 0) {
    std::vector<double> seq_mult = multiplication(A, B);
    ASSERT_EQ(parallel_mult, seq_mult);
  }
}

TEST(Matrix_CCS, Matrix_50x50) {
  int ProcRank;
  matrix_CCS A, B;
  std::vector<double> parallel_mult;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    A = build_data_of_matrix_CCS(random_matrix(50, 50), 50, 50);
    B = build_data_of_matrix_CCS(random_matrix(50, 50), 50, 50);
    parallel_mult = parallel_multiplication(A, B);
  } else {
    A = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    B = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    parallel_mult = parallel_multiplication(A, B);
  }

  if (ProcRank == 0) {
    std::vector<double> seq_mult = multiplication(A, B);
    ASSERT_EQ(parallel_mult, seq_mult);
  }
}

TEST(Matrix_CCS, Col_eq_row) {
  int ProcRank;
  matrix_CCS A, B;
  std::vector<double> parallel_mult;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    A = build_data_of_matrix_CCS(random_matrix(50, 2), 50, 2);
    B = build_data_of_matrix_CCS(random_matrix(2, 32), 2, 32);
    parallel_mult = parallel_multiplication(A, B);
  } else {
    A = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    B = build_data_of_matrix_CCS(random_matrix(1, 1), 1, 1);
    parallel_mult = parallel_multiplication(A, B);
  }

  if (ProcRank == 0) {
    std::vector<double> seq_mult = multiplication(A, B);
    ASSERT_EQ(parallel_mult, seq_mult);
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
