// Copyright 2021 Olenin Sergey
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include "./gradient.h"
#include <gtest-mpi-listener.hpp>

TEST(method_conjugate_gradient,
     throw_size) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 2;
  std::vector<double> matrix(size * size);
  std::vector<double> vector(size + 1);

  if (rank == 0) {
    ASSERT_ANY_THROW(matrix_vec_mult(matrix, vector));
  }
}

TEST(method_conjugate_gradient, correct_vec_mult) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 2;
  std::vector<double> vector1(size);
  std::vector<double> vector2(size);
  double correctAnswer;

  vector1[0] = 1;
  vector1[1] = 2;

  vector2[0] = 5;
  vector2[1] = 6;

  correctAnswer = 17;

  if (rank == 0) {
    double result = vec_mult(vector1, vector2);
    ASSERT_NEAR(correctAnswer, result, 0.5);
  }
}

TEST(method_conjugate_gradient, correct_matrix_vec_mult) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 2;
  std::vector<double> vector(size);
  std::vector<double> matrix(size * size);
  std::vector<double> correctAnswer(size);

  vector[0] = 1;
  vector[1] = 2;

  matrix[0] = 4;
  matrix[1] = 5;
  matrix[2] = 6;
  matrix[3] = 7;

  correctAnswer[0] = 14;
  correctAnswer[1] = 20;

  if (rank == 0) {
    std::vector<double> result = matrix_vec_mult(matrix, vector);
    for (size_t i = 0; i < result.size(); i++)
      ASSERT_NEAR(correctAnswer[i], result[i], 0.5);
  }
}

TEST(method_conjugate_gradient, orrect_get_sol_for_one_proc) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 3;
  std::vector<double> vector(size);
  std::vector<double> matrix(size * size);
  std::vector<double> correctAnswer(size);

  vector[0] = 1;
  vector[1] = 5;
  vector[2] = 9;

  matrix[0] = 3;
  matrix[1] = 4;
  matrix[2] = 0;
  matrix[3] = 4;
  matrix[4] = -3;
  matrix[5] = 0;
  matrix[6] = 0;
  matrix[7] = 0;
  matrix[8] = 5;

  correctAnswer[0] = 0.92;
  correctAnswer[1] = -0.44;
  correctAnswer[2] = 1.80;

  if (rank == 0) {
    std::vector<double> result = get_sol_for_one_proc(matrix, vector, size);
    for (size_t i = 0; i < result.size(); i++)
      ASSERT_NEAR(correctAnswer[i], result[i], 0.5);
  }
}

TEST(method_conjugate_gradient, correct_get_sol_for_multiply_proc) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 12;
  std::vector<double> vector = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  std::vector<double> resultPar =
    get_sol_for_multiply_proc(matrix, vector, size);

  if (rank == 0) {
    std::vector<double> resultSeq =
      get_sol_for_one_proc(matrix, vector, size);
    for (size_t i = 0; i < resultSeq.size(); i++)
      ASSERT_NEAR(resultSeq[i], resultPar[i], 0.5);
  }
}
/*
TEST(method_conjugate_gradient, size_100) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 100;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vec = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vec, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vec, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
TEST(method_conjugate_gradient, size_500) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 500;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vec = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vec, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vec, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
TEST(method_conjugate_gradient, size_1000) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 1000;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vector = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vector, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vector, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
TEST(method_conjugate_gradient, size_1500) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 1500;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vector = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vector, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vector, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
TEST(method_conjugate_gradient, size_2000) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 2000;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vector = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vector, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vector, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
TEST(method_conjugate_gradient, size_2500) {
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 2500;

  double time_start_one, time_finish_one, time_one;
  double time_start_multiply, time_finish_multiply, time_multiply;

  std::vector<double> vector = rand_vec(size);
  std::vector<double> matrix = rand_matrix(size);

  time_start_multiply = MPI_Wtime();

  std::vector<double> x_par = get_sol_for_multiply_proc(matrix, vector, size);

  time_finish_multiply = MPI_Wtime();
  time_multiply = time_finish_multiply - time_start_multiply;

  if (rank == 0) {
    time_start_one = MPI_Wtime();

    std::vector<double> x_solo = get_sol_for_one_proc(matrix, vector, size);

    time_finish_one = MPI_Wtime();
    time_one = time_finish_one - time_start_one;

    std::cout << "one proc time " << time_one << std::endl;
    std::cout << "multiply proc time " << time_multiply << std::endl;

    for (size_t i = 0; i < x_solo.size(); i++)
      ASSERT_NEAR(x_solo[i], x_par[i], 0.5);
  }
}
*/

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
