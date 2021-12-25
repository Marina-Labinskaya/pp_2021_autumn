// Copyright 2021 Moskvin Stanislav
#include <mpi.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <utility>
#include "../../../modules/task_3/moskvin_s_trapezium_method/trapezium_method.h"

double getMethodSequential(double a_x, double b_x, double a_y, double b_y,
                        int n, double (*Func)(double, double)) {
    double sum = 0;
    double dx = (b_x - a_x) / n;
    double dy = (b_y - a_y) / n;
    double zero = pow(10.0, -8);

    for (double x = a_x + dx; std::abs(x - b_x) > zero; x += dx) {
        double cur_sum = (Func(x, a_y) + Func(x, b_y)) / 2;
        for (double y = a_y + dy; std::abs(y - b_y) > zero; y += dy) {
            cur_sum += Func(x, y);
        }
        sum += cur_sum;
    }

    sum += (Func(a_x, a_y) + Func(a_x, b_y) +
                Func(b_x, a_y) + Func(b_x, b_y)) / 4;
    for (double y = a_y + dy; std::abs(y - b_y) > zero; y += dy) {
        sum += Func(a_x, y) / 2;
        sum += Func(b_x, y) / 2;
    }

    return sum * dx * dy;
}

double getMethodParallel(double a_x, double b_x, double a_y, double b_y,
                        int n, double (*Func)(double, double)) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double global;
    double local = 0;
    double dx = (b_x - a_x) / n;
    double dy = (b_y - a_y) / n;
    double zero = pow(10.0, -8);
    double x;

    if (rank == 0) {
        x = a_x + dx * size;
    } else {
        x = a_x + dx * rank;
    }
    while (b_x - x > zero) {
        double cur_sum = (Func(x, a_y) + Func(x, b_y)) / 2;
        for (double y = a_y + dy; std::abs(y - b_y) > zero; y += dy) {
            cur_sum += Func(x, y);
        }
        local += cur_sum;
        x += dx * size;
    }
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        global += (Func(a_x, a_y) + Func(a_x, b_y) +
                    Func(b_x, a_y) + Func(b_x, b_y)) / 4;
        for (double y = a_y + dy; std::abs(y - b_y) > zero; y += dy) {
            global += Func(a_x, y) / 2;
            global += Func(b_x, y) / 2;
        }
    }
    return global * dx * dy;
}
