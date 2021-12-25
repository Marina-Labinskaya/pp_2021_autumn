// Copyright 2021 Labinskaya Marina
#include <mpi.h>
#include <vector>
#include <ctime>
#ifndef MODULES_TASK_3_LABINSKAYA_M_MULT_MATR_CANNON_ALGORITHM_CANNON_ALGORITHM_H_
#define MODULES_TASK_3_LABINSKAYA_M_MULT_MATR_CANNON_ALGORITHM_CANNON_ALGORITHM_H_

double* creating_zero_matrix(int size);
double* creating_random_matrix(int size);
double* sequantial_mult(int size, double* matr1, double* matr2);
MPI_Comm creating_decart_topology(int num);
double* parallel_mult(int size, double* matr1, double* matr2);

#endif  // MODULES_TASK_3_LABINSKAYA_M_MULT_MATR_CANNON_ALGORITHM_CANNON_ALGORITHM_H_
