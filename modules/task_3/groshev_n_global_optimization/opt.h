// Copyright 2021 Groshev Nikolay
#ifndef MODULES_TASK_3_GROSHEV_N_GLOBAL_OPTIMIZATION_OPT_H_
#define MODULES_TASK_3_GROSHEV_N_GLOBAL_OPTIMIZATION_OPT_H_
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

double seq_global_opt(double a, double b, std::function<double(double*)> f,
                      int part, double eps);
double perallel_global_opt(double a, double b, std::function<double(double*)> f,
                           int part, double eps);

#endif  // MODULES_TASK_3_GROSHEV_N_GLOBAL_OPTIMIZATION_OPT_H_
