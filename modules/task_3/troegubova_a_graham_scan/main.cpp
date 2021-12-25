// Copyright 2021 Troegubova Alexandra
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "./graham_scan.h"
#include <gtest-mpi-listener.hpp>

TEST(grahan_scan_test, correct_finding_of_a_convex_hull_on_a_set_of_10_points) {
    int size = 10;
    std::vector<int> ps(0);
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    std::vector<int> shell_par(0);
    int size_shell_par;
    int* p_ssp = &size_shell_par;
    if (proc_rank == 0) {
        ps = GetPoints(size, 600, 350, 330);

        // double time_1 = MPI_Wtime();
        shell_par = GrahamParallel(ps, size, p_ssp);
        // double time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamParallel execution: "<< (time_2 - time_1) <<  "\n";

        int size_shell_seq;
        int* p_sss = &size_shell_seq;
        // time_1 = MPI_Wtime();
        std::vector<int> shell_seq = GrahamSequential(ps, size, p_sss);
        // time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamSequential execution " << (time_2 - time_1) <<  "\n";
        int control = 1;
        if (size_shell_par != size_shell_seq) {
            control = 0;
        } else {
            for (int i = 0; i < size_shell_par; i++) {
                if ((shell_par[i * 2] != shell_seq[i * 2]) || (shell_par[i * 2 + 1] != shell_seq[i * 2 + 1])) {
                    control = 0;
                    break;
                }
            }
        }
        ASSERT_EQ(control, 1);
    } else {
        shell_par = GrahamParallel(ps, size, p_ssp);
    }
}

TEST(grahan_scan_test, correct_finding_of_a_convex_hull_on_a_set_of_30_points) {
    int size = 30;
    std::vector<int> ps(0);
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    std::vector<int> shell_par(0);
    int size_shell_par;
    int* p_ssp = &size_shell_par;
    if (proc_rank == 0) {
        ps = GetPoints(size, 600, 350, 330);

        // double time_1 = MPI_Wtime();
        shell_par = GrahamParallel(ps, size, p_ssp);
        // double time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamParallel execution: "<< (time_2 - time_1) <<  "\n";

        int size_shell_seq;
        int* p_sss = &size_shell_seq;
        // time_1 = MPI_Wtime();
        std::vector<int> shell_seq = GrahamSequential(ps, size, p_sss);
        // time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamSequential execution " << (time_2 - time_1) <<  "\n";
        int control = 1;
        if (size_shell_par != size_shell_seq) {
            control = 0;
        } else {
            for (int i = 0; i < size_shell_par; i++) {
                if ((shell_par[i * 2] != shell_seq[i * 2]) || (shell_par[i * 2 + 1] != shell_seq[i * 2 + 1])) {
                    control = 0;
                    break;
                }
            }
        }
        ASSERT_EQ(control, 1);
    } else {
        shell_par = GrahamParallel(ps, size, p_ssp);
    }
}

TEST(grahan_scan_test, correct_finding_of_a_convex_hull_on_a_set_of_80_points) {
    int size = 80;
    std::vector<int> ps(0);
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    std::vector<int> shell_par(0);
    int size_shell_par;
    int* p_ssp = &size_shell_par;
    if (proc_rank == 0) {
        ps = GetPoints(size, 600, 350, 330);

        // double time_1 = MPI_Wtime();
        shell_par = GrahamParallel(ps, size, p_ssp);
        // double time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamParallel execution: "<< (time_2 - time_1) <<  "\n";

        int size_shell_seq;
        int* p_sss = &size_shell_seq;
        // time_1 = MPI_Wtime();
        std::vector<int> shell_seq = GrahamSequential(ps, size, p_sss);
        // time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamSequential execution " << (time_2 - time_1) <<  "\n";
        int control = 1;
        if (size_shell_par != size_shell_seq) {
            control = 0;
        } else {
            for (int i = 0; i < size_shell_par; i++) {
                if ((shell_par[i * 2] != shell_seq[i * 2]) || (shell_par[i * 2 + 1] != shell_seq[i * 2 + 1])) {
                    control = 0;
                    break;
                }
            }
        }
        ASSERT_EQ(control, 1);
    } else {
        shell_par = GrahamParallel(ps, size, p_ssp);
    }
}

TEST(grahan_scan_test, correct_finding_of_a_convex_hull_on_a_set_of_100_points) {
    int size = 100;
    std::vector<int> ps(0);
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    std::vector<int> shell_par(0);
    int size_shell_par;
    int* p_ssp = &size_shell_par;
    if (proc_rank == 0) {
        ps = GetPoints(size, 600, 350, 330);

        // double time_1 = MPI_Wtime();
        shell_par = GrahamParallel(ps, size, p_ssp);
        // double time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamParallel execution: "<< (time_2 - time_1) <<  "\n";

        int size_shell_seq;
        int* p_sss = &size_shell_seq;
        // time_1 = MPI_Wtime();
        std::vector<int> shell_seq = GrahamSequential(ps, size, p_sss);
        // time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamSequential execution " << (time_2 - time_1) <<  "\n";
        int control = 1;
        if (size_shell_par != size_shell_seq) {
            control = 0;
        } else {
            for (int i = 0; i < size_shell_par; i++) {
                if ((shell_par[i * 2] != shell_seq[i * 2]) || (shell_par[i * 2 + 1] != shell_seq[i * 2 + 1])) {
                    control = 0;
                    break;
                }
            }
        }
        ASSERT_EQ(control, 1);
    } else {
        shell_par = GrahamParallel(ps, size, p_ssp);
    }
}

TEST(grahan_scan_test, correct_finding_of_a_convex_hull_on_a_set_of_200_points) {
    int size = 200;
    std::vector<int> ps(0);
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    std::vector<int> shell_par(0);
    int size_shell_par;
    int* p_ssp = &size_shell_par;
    if (proc_rank == 0) {
        ps = GetPoints(size, 600, 350, 330);

        // double time_1 = MPI_Wtime();
        shell_par = GrahamParallel(ps, size, p_ssp);
        // double time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamParallel execution: "<< (time_2 - time_1) <<  "\n";

        int size_shell_seq;
        int* p_sss = &size_shell_seq;
        // time_1 = MPI_Wtime();
        std::vector<int> shell_seq = GrahamSequential(ps, size, p_sss);
        // time_2 = MPI_Wtime();
        // std::cout << "Time spent on GrahamSequential execution " << (time_2 - time_1) <<  "\n";
        int control = 1;
        if (size_shell_par != size_shell_seq) {
            control = 0;
        } else {
            for (int i = 0; i < size_shell_par; i++) {
                if ((shell_par[i * 2] != shell_seq[i * 2]) || (shell_par[i * 2 + 1] != shell_seq[i * 2 + 1])) {
                    control = 0;
                    break;
                }
            }
        }
        ASSERT_EQ(control, 1);
    } else {
        shell_par = GrahamParallel(ps, size, p_ssp);
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
