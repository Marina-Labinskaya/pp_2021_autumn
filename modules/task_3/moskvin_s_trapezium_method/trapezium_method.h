// Copyright 2021 Moskvin Stanislav
#ifndef MODULES_TASK_3_MOSKVIN_S_TRAPEZIUM_METHOD_TRAPEZIUM_METHOD_H_
#define MODULES_TASK_3_MOSKVIN_S_TRAPEZIUM_METHOD_TRAPEZIUM_METHOD_H_

double getMethodSequential(double a_x, double b_x, double a_y, double b_y,
                                int n, double (*Func)(double, double));
double getMethodParallel(double a_x, double b_x, double a_y, double b_y,
                                int n, double (*Func)(double, double));

#endif  //  MODULES_TASK_3_MOSKVIN_S_TRAPEZIUM_METHOD_TRAPEZIUM_METHOD_H_
