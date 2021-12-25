// Copyright 2021 Tyrina Anastasia
#include "../../../modules/task_3/tyrina_a_simpson/simpson.h"

#include <mpi.h>

#include <cmath>

double getSequentialSimpson(function<double(vector<double>)> func,
                            vector<pair<double, double>> a_b, int n) {
  int integral_size = a_b.size();
  vector<double> h(integral_size);
  int num = 1;

  for (int i = 0; i < integral_size; ++i) {
    h[i] = (a_b[i].second - a_b[i].first) / n;
    num *= n;
  }

  double sum = 0.0;
  for (int i = 0; i < num; ++i) {
    vector<vector<double>> x(integral_size);

    int tmp = i;
    for (int j = 0; j < integral_size; ++j) {
      x[j].push_back(a_b[j].first + tmp % n * h[j]);

      for (int k = 0; k < 4; k++) {
        x[j].push_back(a_b[j].first + tmp % n * h[j] + h[j] / 2);
      }
      x[j].push_back(a_b[j].first + tmp % n * h[j] + h[j]);
      tmp /= n;
    }
    vector<double> comb;
    for (int i = 0; i < pow(6, integral_size); ++i) {
      int tmp = i;
      for (int j = 0; j < integral_size; ++j) {
        comb.push_back(x[j][tmp % 6]);
        tmp /= 6;
      }
      sum += func(comb);
      comb.clear();
    }
    x.clear();
  }
  for (int i = 0; i < integral_size; ++i) {
    sum *= h[i] / 6.0;
  }
  return sum;
}

double getParallelSimpson(function<double(vector<double>)> func,
                          vector<pair<double, double>> a_b, int n) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int integral_size = a_b.size();
  std::vector<double> h(integral_size);

  int global_num;
  if (rank == 0) {
    global_num = 1;
    for (int i = 0; i < integral_size; ++i) {
      h[i] = (a_b[i].second - a_b[i].first) / n;
      global_num *= n;
    }
  }
  MPI_Bcast(&global_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(h.data(), integral_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int rem = global_num % size;
  int delta, start;
  if (rank == 0) {
    delta = global_num / size + rem;
    start = 0;
  } else {
    delta = global_num / size;
    start = rem + delta * rank;
  }

  double local_sum = 0.0;
  for (int i = start; i < delta + start; ++i) {
    vector<vector<double>> x(integral_size);
    int tmp = i;
    for (int j = 0; j < integral_size; ++j) {
      x[j].push_back(a_b[j].first + tmp % n * h[j]);
      for (int k = 0; k < 4; k++) {
        x[j].push_back(a_b[j].first + tmp % n * h[j] + h[j] / 2);
      }
      x[j].push_back(a_b[j].first + tmp % n * h[j] + h[j]);
      tmp /= n;
    }

    vector<double> comb;
    for (int i = 0; i < pow(6, integral_size); ++i) {
      int temp = i;
      for (int j = 0; j < integral_size; ++j) {
        comb.push_back(x[j][temp % 6]);
        temp /= 6;
      }
      local_sum += func(comb);
      comb.clear();
    }
    x.clear();
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < integral_size; ++i) {
      global_sum *= h[i] / 6.0;
    }
  }
  return global_sum;
}
