// Copyright 2021 Yashina Darya
#ifndef MODULES_TASK_3_YASHINA_D_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_
#define MODULES_TASK_3_YASHINA_D_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_

#include <cmath>

struct elemOneDim {
  double x;
  double y;
  elemOneDim(double x, double y) : x(x), y(y) {}
  friend bool operator<(const elemOneDim& l, const elemOneDim& r) {
    return l.x < r.x;
  }
};

struct elemTwoDim {
  double x;
  double y;
  double z;
  elemTwoDim(double x, double y, double z = 0) : x(x), y(y), z(z) {}
  friend bool operator<(const elemTwoDim& l, const elemTwoDim& r) {
    return l.x < r.x;
  }
};

struct elemR {
  double R;
  double x;
  double z;
  double x_prev;
  double z_prev;
  elemR(double R, double x, double z, double x_prev, double z_prev)
      : R(R), x(x), z(z), x_prev(x_prev), z_prev(z_prev) {}
  friend bool operator<(const elemR& l, const elemR& r) { return l.R > r.R; }
};

struct threeDimRes {
  double x;
  double y;
  double z;
};

bool isCorrect(threeDimRes a, threeDimRes b, double eps);

threeDimRes calcOneDim(double a, double b, double x,
                        double (*func)(double x, double y), double eps = 0.1,
                        int max_iterations = 100, double r = 2.0);

threeDimRes getGlobalOptimizationParallel(double a1, double b1, double a2, double b2,
                       double (*func)(double x, double y), double eps = 0.1,
                       int max_iterations = 100, double eps_one = 0.1,
                       int max_iterations_one = 100, double r = 2.0);

threeDimRes getGlobalOptimizationSequential(
    double a1, double b1, double a2, double b2,
    double (*func)(double x, double y), double eps = 0.1,
    int max_iterations = 100, double eps_one = 0.1,
    int max_iterations_one = 100, double r = 2.0);

#endif  // MODULES_TASK_3_YASHINA_D_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_
