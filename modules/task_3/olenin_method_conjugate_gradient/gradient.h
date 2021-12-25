// Copyright 2021 Olenin Sergey
#ifndef MODULES_TASK_3_OLENIN_METHOD_CONJUGATE_GRADIENT_GRADIENT_H_
#define MODULES_TASK_3_OLENIN_METHOD_CONJUGATE_GRADIENT_GRADIENT_H_
#include <vector>

std::vector<double> rand_vec(int size);

double vec_mult(const std::vector<double>& vec_a,
                const std::vector<double>& vec_b);


std::vector<double> rand_matrix(int size);

std::vector<double> matrix_vec_mult(const std::vector<double>& matrix,
                                    const std::vector<double>& vec);

std::vector<double> get_sol_for_one_proc(const std::vector<double>& matrix,
                                const std::vector<double>& vec, int size);

std::vector<double> get_sol_for_multiply_proc(const std::vector<double>& matrix,
                                const std::vector<double>& vec,
                                int size);
#endif  // MODULES_TASK_3_OLENIN_METHOD_CONJUGATE_GRADIENT_GRADIENT_H_
