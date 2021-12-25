// Copyright 2021 Yashina Darya
#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <gtest-mpi-listener.hpp>

#include "./global_optimization.h"

double func1(double x, double y) { return std::pow(x, 2) + std::pow(y - 1, 2); }

double func2(double x, double y) {
  return 4 + std::pow(std::pow(std::pow(x, 2) + std::pow(y, 2), 2), 1.0 / 3);
}

double func3(double x, double y) {
  return std::pow(x, 3) + 8 * std::pow(y, 3) - 6 * x * y + 5;
}
double func4(double x, double y) {
  return y * sqrt(x) - 2 * std::pow(y, 2) - x + 14 * y;
}
double func5(double x, double y) {
  return x + 4 * y - 6 - 2 * log(x * y) - 3 * log(y);
}

TEST(GLOBAL_OPTIMIZATION_PARALLEL, TEST_1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double (*function_pointer)(double, double) = func1;
  double start = MPI_Wtime();
  threeDimRes result =
      getGlobalOptimizationParallel(-10, 5, -5, 5, function_pointer);
  double end = MPI_Wtime();
  if (rank == 0) {
    double ptime = end - start;

    start = MPI_Wtime();
    threeDimRes reference_result =
        getGlobalOptimizationSequential(-10, 5, -5, 5, function_pointer);
    end = MPI_Wtime();
    double stime = end - start;

    std::cout << stime / ptime << std::endl;

    ASSERT_TRUE(isCorrect(result, reference_result, 0.1));
  }
}

TEST(GLOBAL_OPTIMIZATION_PARALLEL, TEST_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double (*function_pointer)(double, double) = func2;
  threeDimRes result =
      getGlobalOptimizationParallel(-0.9, 1, -1, 1, function_pointer);
  if (rank == 0) {
    threeDimRes reference_result = {0, 0, 4};
    ASSERT_TRUE(isCorrect(result, reference_result, 0.1));
  }
}

TEST(GLOBAL_OPTIMIZATION_PARALLEL, TEST_3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double (*function_pointer)(double, double) = func3;
  threeDimRes result =
      getGlobalOptimizationParallel(0, 2, 0.3, 1, function_pointer);
  if (rank == 0) {
    threeDimRes reference_result = {1, 0.5, 4};
    ASSERT_TRUE(isCorrect(result, reference_result, 0.1));
  }
}

TEST(GLOBAL_OPTIMIZATION_PARALLEL, TEST_4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double (*function_pointer)(double, double) = func4;
  threeDimRes result =
      getGlobalOptimizationParallel(0.1, 5, -5, 5, function_pointer);
  if (rank == 0) {
    threeDimRes reference_result = {5, -5, -136.18};
    ASSERT_TRUE(isCorrect(result, reference_result, 0.1));
  }
}

TEST(GLOBAL_OPTIMIZATION_PARALLEL, TEST_5) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double (*function_pointer)(double, double) = func5;
  threeDimRes result =
      getGlobalOptimizationParallel(1, 3, 1, 2, function_pointer);
  if (rank == 0) {
    threeDimRes reference_result = {2, 1.25, -1.5};
    ASSERT_TRUE(isCorrect(result, reference_result, 0.1));
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
