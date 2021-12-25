// Copyright 2021 Lazarev Alexey

#ifndef MODULES_TASK_3_LAZAREV_A_CCS_CCS_CCS_H_
#define MODULES_TASK_3_LAZAREV_A_CCS_CCS_CCS_H_

#include <mpi.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

struct matrix_CCS {
  int rows, cols, non_zero;
  std::vector<double> value;
  std::vector<int> Rows;
  std::vector<int> ColumnIndexes;

  friend std::vector<double> multiplication(const matrix_CCS& A,
                                            const matrix_CCS& B);
};

std::vector<double> random_matrix(int rows, int cols);
matrix_CCS build_data_of_matrix_CCS(const std::vector<double>& _matrix,
                                    const int& _rows, const int& _cols);
std::vector<double> parallel_multiplication(const matrix_CCS& A,
                                            const matrix_CCS& B);

#endif  // MODULES_TASK_3_LAZAREV_A_CCS_CCS_CCS_H_
