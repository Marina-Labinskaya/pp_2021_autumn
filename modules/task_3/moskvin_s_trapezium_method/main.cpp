// Copyright 2021 Moskvin Stanislav
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "../../../modules/task_3/moskvin_s_trapezium_method/trapezium_method.h"
#include <gtest-mpi-listener.hpp>

TEST(parallel_operations_mpi, integr_trapezoid_par_seq_function_xsiny) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time_start, time_end, par_time, seq_time, res, seq_res;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    auto Func = [](double x, double y) -> double {
        return x * sin(y);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    res = getMethodParallel(0, 5, 0, 1, 100, Func);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double error = pow(10.0, -8);
        time_end = MPI_Wtime();
        par_time = time_end - time_start;
        time_start = MPI_Wtime();
        seq_res = getMethodSequential(0, 5, 0, 1, 100, Func);
        time_end = MPI_Wtime();
        seq_time = time_end - time_start;
        std::cout << seq_time / par_time << "\n";
        ASSERT_NEAR(res, seq_res, error);
    }
}

TEST(parallel_operations_mpi, integr_trapezoid_par_seq_function_xy) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time_start, time_end, par_time, seq_time, res, seq_res;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    auto Func = [](double x, double y) -> double {
        return x * y;
    };

    MPI_Barrier(MPI_COMM_WORLD);
    res = getMethodParallel(0, 1, 0, 5, 1000, Func);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double error = pow(10.0, -8);
        time_end = MPI_Wtime();
        par_time = time_end - time_start;
        time_start = MPI_Wtime();
        seq_res = getMethodSequential(0, 1, 0, 5, 1000, Func);
        time_end = MPI_Wtime();
        seq_time = time_end - time_start;
        std::cout << seq_time / par_time << "\n";
        ASSERT_NEAR(res, seq_res, error);
    }
}

TEST(parallel_operations_mpi, integr_trapezoid_par_seq_function_pow) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time_start, time_end, par_time, seq_time, res, seq_res;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    auto Func = [](double x, double y) -> double {
        return pow(x, y);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    res = getMethodParallel(0, 0.5, 0, 0.5, 1000, Func);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double error = pow(10.0, -8);
        time_end = MPI_Wtime();
        par_time = time_end - time_start;
        time_start = MPI_Wtime();
        seq_res = getMethodSequential(0, 0.5, 0, 0.5, 1000, Func);
        time_end = MPI_Wtime();
        seq_time = time_end - time_start;
        std::cout << seq_time / par_time << "\n";
        ASSERT_NEAR(res, seq_res, error);
    }
}

TEST(parallel_operations_mpi, integr_trapezoid_par_seq_function_0) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time_start, time_end, par_time, seq_time, res, seq_res;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    auto Func = [](double x, double y) -> double {
        return 0;
    };

    MPI_Barrier(MPI_COMM_WORLD);
    res = getMethodParallel(0, 50, 0, 50, 10000, Func);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double error = pow(10.0, -8);
        time_end = MPI_Wtime();
        par_time = time_end - time_start;
        time_start = MPI_Wtime();
        seq_res = getMethodSequential(0, 50, 0, 50, 10000, Func);
        time_end = MPI_Wtime();
        seq_time = time_end - time_start;
        std::cout << seq_time / par_time << "\n";
        ASSERT_NEAR(res, seq_res, error);
    }
}

TEST(parallel_operations_mpi, integr_trapezoid_par_seq_function_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time_start, time_end, par_time, seq_time, res, seq_res;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    auto Func = [](double x, double y) -> double {
        return 1;
    };

    MPI_Barrier(MPI_COMM_WORLD);
    res = getMethodParallel(0, 5, 0, 5, 1000, Func);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double error = pow(10.0, -8);
        time_end = MPI_Wtime();
        par_time = time_end - time_start;
        time_start = MPI_Wtime();
        seq_res = getMethodSequential(0, 5, 0, 5, 1000, Func);
        time_end = MPI_Wtime();
        seq_time = time_end - time_start;
        std::cout << seq_time / par_time << "\n";
        ASSERT_NEAR(res, seq_res, error);
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
