// Copyright 2021 Tyrina Anastasia
#include <gtest/gtest.h>

#include <cmath>
#include <gtest-mpi-listener.hpp>

#include "./simpson.h"

const function<double(vector<double>)> func1 = [](vector<double> vec) {
  double x = vec[0];
  double y = vec[1];
  return x * x - 2 * y * y;
};

const function<double(vector<double>)> func2 = [](vector<double> vec) {
  double x = vec[0];
  double y = vec[1];
  return sin(x) + 10 * y * y;
};

const function<double(vector<double>)> func3 = [](vector<double> vec) {
  double x = vec[0];
  double y = vec[1];
  return sqrt(x) * sin(x) - 2 * cos(y * y);
};

const function<double(vector<double>)> func4 = [](vector<double> vec) {
  double x = vec[0];
  double y = vec[1];
  double z = vec[2];
  return sin(x * x) - 2 * y - cos(z);
};

const function<double(vector<double>)> func5 = [](vector<double> vec) {
  double x = vec[0];
  double y = vec[1];
  double z = vec[2];
  return x * z - 2 * y + y * y - cos(z);
};

const double eps = 0.0001;

TEST(SIMPSON_METHOD_MPI, TEST_1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> a_b({{4, 10}, {1, 2}});
  int n = 70;

  double start = MPI_Wtime();
  double result = getParallelSimpson(func1, a_b, n);
  double end = MPI_Wtime();

  if (rank == 0) {
    double ptime = end - start;

    start = MPI_Wtime();
    double reference_result = getSequentialSimpson(func1, a_b, n);
    end = MPI_Wtime();
    double stime = end - start;

    std::cout << "Sequential: " << stime << std::endl;
    std::cout << "Parallel: " << ptime << std::endl;
    std::cout << "Speedup: " << stime / ptime << std::endl;
    ASSERT_NEAR(result, reference_result, eps);
  }
}

TEST(SIMPSON_METHOD_MPI, TEST_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> a_b({{1, 3}, {1, 2}});
  int n = 60;

  double result = getParallelSimpson(func2, a_b, n);

  if (rank == 0) {
    double reference_result = getSequentialSimpson(func2, a_b, n);

    ASSERT_NEAR(result, reference_result, eps);
  }
}

TEST(SIMPSON_METHOD_MPI, TEST_3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> a_b({{1, 5}, {3, 4}});
  int n = 100;

  double result = getParallelSimpson(func3, a_b, n);

  if (rank == 0) {
    double reference_result = getSequentialSimpson(func3, a_b, n);

    ASSERT_NEAR(result, reference_result, eps);
  }
}

TEST(SIMPSON_METHOD_MPI, TEST_4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> a_b({{1, 3}, {5, 6}, {3, 7}});
  int n = 10;

  double result = getParallelSimpson(func4, a_b, n);

  if (rank == 0) {
    double reference_result = getSequentialSimpson(func4, a_b, n);

    ASSERT_NEAR(result, reference_result, eps);
  }
}

TEST(SIMPSON_METHOD_MPI, TEST_5) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::pair<double, double>> a_b({{4, 10}, {1, 2}, {3, 5}});
  int n = 10;

  double start = MPI_Wtime();
  double result = getParallelSimpson(func5, a_b, n);
  double end = MPI_Wtime();

  if (rank == 0) {
    double ptime = end - start;

    start = MPI_Wtime();
    double reference_result = getSequentialSimpson(func5, a_b, n);
    end = MPI_Wtime();
    double stime = end - start;

    std::cout << "Sequential: " << stime << std::endl;
    std::cout << "Parallel: " << ptime << std::endl;
    std::cout << "Speedup: " << stime / ptime << std::endl;
    ASSERT_NEAR(result, reference_result, eps);
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
