// Copyright 2021 Labinskaya Marina
#include <gtest/gtest.h>
#include <gtest-mpi-listener.hpp>
#include "./cannon_algorithm.h"

TEST(TEST_RANDOM, can_create_random_matrix) {
    int size = 10;
    ASSERT_NO_THROW(creating_random_matrix(size));
}

TEST(SEQUANTIAL_MULT, compute_sequantial_mult_correctly) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (ProcRank == 0) {
        int size = 3;
        double* matr1 = creating_zero_matrix(size);
        double* matr2 = creating_zero_matrix(size);
        for (int i = 0; i < size * size; ++i) {
            matr1[i] = i + 1;
            matr2[i] = 2 * i;
        }
        double correct_res[] = { 48, 60, 72, 102, 132, 162, 156, 204, 252};
        double* res = sequantial_mult(size, matr1, matr2);

        for (int i = 0; i < size * size; i++) {
            ASSERT_NEAR(correct_res[i], res[i], 1e-5);
        }
    }
}

TEST(PARALLEL_MULT, compute_parallel_mult_correctly) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    int size = 3;
    double* matr1 = nullptr;
    double* matr2 = nullptr;
    if (ProcRank == 0) {
        matr1 = creating_zero_matrix(size);
        matr2 = creating_zero_matrix(size);
        for (int i = 0; i < size * size; ++i) {
            matr1[i] = i + 1;
            matr2[i] = 2 * i;
        }
    }
    double* res = parallel_mult(size, matr1, matr2);
    if (ProcRank == 0) {
        double correct_res[] = { 48, 60, 72, 102, 132, 162, 156, 204, 252 };
        for (int i = 0; i < size * size; i++) {
            ASSERT_NEAR(correct_res[i], res[i], 1e-5);
        }
    }
}

TEST(PARALLEL_MULT, compute_parallel_mult_correctly_with_random_matrix) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    int size = 10;
    double* matr1 = nullptr;
    double* matr2 = nullptr;
    double* seq_res = nullptr;

    double t_st, t_end, seq_t, parallel_t;
    if (ProcRank == 0) {
        matr1 = creating_random_matrix(size);
        matr2 = creating_random_matrix(size);
        t_st = MPI_Wtime();
        seq_res = sequantial_mult(size, matr1, matr2);
        t_end = MPI_Wtime();
        seq_t = t_end - t_st;
	}

    if (ProcRank == 0) {
		t_st = MPI_Wtime();
    }
    double* parallel_res = parallel_mult(size, matr1, matr2);
    if (ProcRank == 0) {
        t_end = MPI_Wtime();
        parallel_t = t_end - t_st;
        for (int i = 0; i < size * size; i++) {
            ASSERT_NEAR(seq_res[i], parallel_res[i], 1e-5);
        }
        std::cout << "Sequantial time = " << seq_t << std::endl;
        std::cout << "Parallel time = " << parallel_t << std::endl;
    }
}

TEST(PARALLEL_MULT, compute_parallel_mult_correctly_with_large_size_random_matrix) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    int size = 500;
    double* matr1 = nullptr;
    double* matr2 = nullptr;
    double* seq_res = nullptr;

    double t_st, t_end, seq_t, parallel_t;
    if (ProcRank == 0) {
        matr1 = creating_random_matrix(size);
        matr2 = creating_random_matrix(size);
        t_st = MPI_Wtime();
        seq_res = sequantial_mult(size, matr1, matr2);
        t_end = MPI_Wtime();
        seq_t = t_end - t_st;
    }

    if (ProcRank == 0) {
        t_st = MPI_Wtime();
    }
    double* parallel_res = parallel_mult(size, matr1, matr2);
    if (ProcRank == 0) {
        t_end = MPI_Wtime();
        parallel_t = t_end - t_st;
        for (int i = 0; i < size * size; i++) {
            ASSERT_NEAR(seq_res[i], parallel_res[i], 1e-5);
        }
        std::cout << "Sequantial time = " << seq_t << std::endl;
        std::cout << "Parallel time = " << parallel_t << std::endl;
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
