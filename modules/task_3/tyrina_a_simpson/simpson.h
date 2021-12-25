// Copyright 2021 Tyrina Anastasia
#ifndef MODULES_TASK_3_TYRINA_A_SIMPSON_SIMPSON_H_
#define MODULES_TASK_3_TYRINA_A_SIMPSON_SIMPSON_H_

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

using std::function;
using std::pair;
using std::vector;

double getSequentialSimpson(function<double(vector<double>)> func,
                            vector<pair<double, double>> a_b, int n);

double getParallelSimpson(function<double(vector<double>)> f,
                          vector<pair<double, double>> limits, int n);

#endif  // MODULES_TASK_3_TYRINA_A_SIMPSON_SIMPSON_H_
