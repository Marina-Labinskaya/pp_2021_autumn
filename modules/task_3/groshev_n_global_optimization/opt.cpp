// Copyright 2021 Groshev Nikolay

#include "../../../modules/task_3/groshev_n_global_optimization/opt.h"

double seq_global_opt(double a, double b,
    std::function<double(double*)> f, int part, double eps) {
  std::vector<double> x;
  int num_max_R = 1;
  double r = 2;
  double R;

  x.push_back(a);
  x.push_back(b);
  double M = std::abs(f(&x[1]) - f(&x[0])) / (x[1] - x[0]);
  double m;
  if (M > 0) {
    m = r * M;
  } else {
    m = 1;
  }

  x.push_back((x[1] + x[0]) / 2 - (f(&x[1]) - f(&x[0])) / (2 * m));
  int count = x.size() - 1;
  while (count < part) {
    sort(x.begin(), x.begin() + x.size());

    M = std::abs(f(&x[1]) - f(&x[0])) / (x[1] - x[0]);
    for (int i = 2; i <= count; i++) {
      M = std::max(M, std::abs(f(&x[i]) - f(&x[i - 1])) / (x[i] - x[i - 1]));
    }
    if (M > 0) {
      m = r * M;
    } else {
      m = 1;
    }

    R = m * (x[1] - x[0]) +
        pow((f(&x[1]) - f(&x[0])), 2) / (m * (x[1] - x[0])) -
        2 * (f(&x[1]) + f(&x[0]));
    num_max_R = 1;
    for (int i = 2; i <= count; ++i) {
      double max_R =
          m * (x[i] - x[i - 1]) +
          std::pow((f(&x[i]) - f(&x[i - 1])), 2) / (m * (x[i] - x[i - 1])) -
          2 * (f(&x[i]) + f(&x[i - 1]));
      if (R < max_R) {
        R = max_R;
        num_max_R = i;
      }
    }

    x.push_back((x[num_max_R] + x[num_max_R - 1]) / 2 -
                (f(&x[num_max_R]) - f(&x[num_max_R - 1])) / (2 * m));
    count++;
    if (x[num_max_R] - x[num_max_R - 1] < eps) {
      break;
    }
  }
  return x[num_max_R];
}

double perallel_global_opt(double a, double b, std::function<double(double*)> f,
                           int part, double eps) {
  int ProcNum;
  int ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcNum == 1) {
    return seq_global_opt(a, b, f, part, eps);
  }
  std::vector<double> x(ProcNum);
  double left = a + ((b - a) / ProcNum) * ProcRank;
  double right = left + (b - a) / ProcNum;

  double local_opt = seq_global_opt(left, right, f, part, eps);
  MPI_Gather(&local_opt, 1, MPI_DOUBLE, &x[0], 1, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (ProcRank == 0) {
    double root = f(&x[0]);
    for (int i = 1; i < ProcNum; ++i) {
      if (f(&x[0] + i) < root) {
        double size_t = x[i];
        x[i] = x[0];
        x[0] = size_t;
        root = f(&x[0] + i);
      }
    }
  }
  return x[0];
}
