// Copyright 2021 Kuklin Andrey
#ifndef MODULES_TASK_3_KUKLIN_A_TRAPEZOID_INTEGRAL_TRAPEZOID_INTEGRAL_H_
#define MODULES_TASK_3_KUKLIN_A_TRAPEZOID_INTEGRAL_TRAPEZOID_INTEGRAL_H_

#include <vector>
#include <cmath>

class Func {
 protected:
  std::vector<double> lower_bound, upper_bound;

 public:
  virtual double CalculateFunc(std::vector<double> point) = 0;
  virtual std::vector<double> GetLowerBound();
  virtual std::vector<double> GetUpperBound();
};

class PiIntegral : public Func {
 public:
  PiIntegral();
  double CalculateFunc(std::vector<double> point);
};

class FirstIntegral : public Func {
 public:
  FirstIntegral();
  double CalculateFunc(std::vector<double> point);
};

class SecondIntegral : public Func {
 public:
  SecondIntegral();
  double CalculateFunc(std::vector<double> point);
};

class ThirdIntegral : public Func {
 public:
  ThirdIntegral();
  double CalculateFunc(std::vector<double> point);
};

double CalculateTrapezoidIntegral(Func* func,
                                  const std::vector<double>& lower_bound,
                                  const std::vector<double>& upper_bound,
                                  std::size_t segments_count,
                                  std::size_t f_point, std::size_t l_point);

double TrapezoidIntegralSequential(Func* func,
                                   const std::vector<double>& lower_bound,
                                   const std::vector<double>& upper_bound,
                                   std::size_t segments_count);

double TrapezoidIntegralParallel(Func* func,
                                 const std::vector<double>& lower_bound,
                                 const std::vector<double>& upper_bound,
                                 std::size_t segments_count);

#endif  //  MODULES_TASK_3_KUKLIN_A_TRAPEZOID_INTEGRAL_TRAPEZOID_INTEGRAL_H_
