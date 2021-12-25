// Copyright 2021 Yashina Darya
#include "../../../modules/task_3/yashina_d_global_optimization/global_optimization.h"

#include <mpi.h>

#include <climits>
#include <random>
#include <set>

threeDimRes calcOneDim(double a, double b, double x,
                       double (*func)(double x, double y), double eps,
                       int max_iterations, double r) {
  bool flag = false;
  std::set<elemOneDim> set;
  threeDimRes result;
  result.x = x;

  double M, current_M, m, R, current_R, new_x;
  double temp = func(x, a);

  set.insert(elemOneDim(a, temp));
  result.y = a;
  result.z = temp;

  temp = func(x, b);
  set.insert(elemOneDim(b, temp));

  if (result.z > temp) {
    result.y = b;
    result.z = temp;
  }

  int iteration = 2;
  auto max_R_iterator = set.begin();
  auto max_prev_R_iterator = set.begin();
  while (!flag && iteration < max_iterations) {
    M = -1;
    auto iterator = set.begin();
    iterator++;
    auto prev_iterator = set.begin();
    while (iterator != set.end()) {
      current_M = std::abs(static_cast<double>(
          (iterator->y - prev_iterator->y) / (iterator->x - prev_iterator->x)));
      if (current_M > M) M = current_M;
      iterator++;
      prev_iterator++;
    }

    if (M > 0) {
      m = r * M;
    } else {
      m = 1;
    }

    iterator = set.begin();
    iterator++;
    prev_iterator = set.begin();

    R = INT_MIN;

    while (iterator != set.end()) {
      current_R = m * (iterator->x - prev_iterator->x) +
                  (std::pow((iterator->y - prev_iterator->y), 2) /
                   (m * (iterator->x - prev_iterator->x))) -
                  2 * (iterator->y - prev_iterator->y);
      if (current_R > R) {
        R = current_R;
        max_R_iterator = iterator;
        max_prev_R_iterator = prev_iterator;
      }
      iterator++;
      prev_iterator++;
    }

    iteration++;

    new_x = (0.5) * (max_R_iterator->x + max_prev_R_iterator->x) -
            ((max_R_iterator->y - max_prev_R_iterator->y) / (2 * m));
    temp = func(x, new_x);
    set.insert(elemOneDim(new_x, temp));

    if (result.z > temp) {
      result.y = new_x;
      result.z = temp;
    }
    if (max_R_iterator->x - max_prev_R_iterator->x <= eps) {
      flag = true;
    }
  }
  return result;
}

threeDimRes getGlobalOptimizationParallel(double a1, double b1, double a2,
                                          double b2,
                                          double (*func)(double x, double y),
                                          double eps, int max_iterations,
                                          double eps_one,
                                          int max_iterations_one, double r) {
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return getGlobalOptimizationSequential(a1, b1, a2, b2, func);
  }

  threeDimRes global_result = {0, 0, 0};
  threeDimRes result;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::set<elemTwoDim> set;

    int iteration;
    if (std::abs(a1 - b1) <= 0.0001 || std::abs(a2 - b2) <= 0.0001) {
      iteration = size + 1;
    } else {
      iteration = size;
    }

    double delta = (b1 - a1) / (iteration - 1);

    for (int i = 0; i < size - 1; ++i) {
      double x = a1 + i * delta;
      MPI_Send(&x, 1, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
    }
    result = calcOneDim(a2, b2, b1, func, eps_one);
    set.insert(elemTwoDim(result.x, result.y, result.z));
    global_result = result;

    if (iteration != size) {
      result = calcOneDim(a2, b2, a1 + delta * size, func, eps_one);
      set.insert(elemTwoDim(result.x, result.y, result.z));
      if (result.z < global_result.z) {
        global_result = result;
      }
    }

    for (int i = 0; i < size - 1; ++i) {
      MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,
               MPI_STATUSES_IGNORE);
      if (result.z < global_result.z) global_result = result;
      set.insert(elemTwoDim(result.x, result.y, result.z));
    }

    std::set<elemR> set_R;
    double M, current_M, m, current_R, new_x;

    bool flag = false;
    while (!flag && iteration < max_iterations) {
      set_R.clear();
      M = -1;
      auto iterator = set.begin();
      iterator++;
      auto prev_iterator = set.begin();

      while (iterator != set.end()) {
        current_M =
            std::abs(static_cast<double>((iterator->z - prev_iterator->z) /
                                         (iterator->x - prev_iterator->x)));
        if (current_M > M) M = current_M;
        iterator++;
        prev_iterator++;
      }

      if (M > 0) {
        m = r * M;
      } else {
        m = 1;
      }

      iterator = set.begin();
      iterator++;
      prev_iterator = set.begin();

      while (iterator != set.end()) {
        current_R = m * (iterator->x - prev_iterator->x) +
                    (std::pow((iterator->z - prev_iterator->z), 2) /
                     (m * (iterator->x - prev_iterator->x))) -
                    2 * (iterator->z - prev_iterator->z);
        set_R.insert(elemR(current_R, iterator->x, iterator->z,
                           prev_iterator->x, prev_iterator->z));
        iterator++;
        prev_iterator++;
      }

      auto iterator_R = set_R.begin();
      for (int i = 0; i < size - 1; ++i) {
        iteration++;
        new_x = (0.5) * (iterator_R->x + iterator_R->x_prev) -
                ((iterator_R->z - iterator_R->z_prev) / (2 * m));
        MPI_Send(&new_x, 1, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
        if (iterator_R->x - iterator_R->x_prev <= eps) {
          flag = true;
        }
        iterator_R++;
      }

      for (int i = 0; i < size - 1; ++i) {
        MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);
        if (result.z < global_result.z) global_result = result;
        set.insert(elemTwoDim(result.x, result.y, result.z));
      }
    }

    for (int i = 0; i < size - 1; ++i) {
      double flag = eps * 0.001;
      MPI_Send(&flag, 1, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
    }
  } else {
    bool flag = false;
    while (!flag) {
      double x;
      MPI_Recv(&x, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      if (x == eps * 0.001) {
        flag = true;
      } else {
        result = calcOneDim(a2, b2, x, func, eps_one, max_iterations_one);
        MPI_Send(&result, 3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      }
    }
  }
  return global_result;
}

threeDimRes getGlobalOptimizationSequential(double a1, double b1, double a2,
                                            double b2,
                                            double (*func)(double x, double y),
                                            double eps, int max_iterations,
                                            double eps_one,
                                            int max_iterations_one, double r) {
  threeDimRes global_result = {0, 0, 0};
  threeDimRes result;
  std::set<elemTwoDim> set;

  result = calcOneDim(a2, b2, a1, func, eps_one, max_iterations_one);
  set.insert(elemTwoDim(result.x, result.y, result.z));
  global_result = result;
  result = calcOneDim(a2, b2, b1, func, eps_one, max_iterations_one);
  set.insert(elemTwoDim(result.x, result.y, result.z));

  if (result.z < global_result.z) {
    global_result = result;
  }

  double M, current_M, m, current_R, new_x;
  int iteration = 2;

  std::set<elemR> set_R;

  bool flag = false;
  while (!flag && iteration < max_iterations) {
    set_R.clear();
    M = -1;

    auto iterator = set.begin();
    iterator++;
    auto prev_iterator = set.begin();

    while (iterator != set.end()) {
      current_M = std::abs(static_cast<double>(
          (iterator->z - prev_iterator->z) / (iterator->x - prev_iterator->x)));
      if (current_M > M) M = current_M;
      iterator++;
      prev_iterator++;
    }

    if (M > 0) {
      m = r * M;
    } else {
      m = 1;
    }

    iterator = set.begin();
    iterator++;
    prev_iterator = set.begin();

    while (iterator != set.end()) {
      current_R = m * (iterator->x - prev_iterator->x) +
                  (std::pow((iterator->z - prev_iterator->z), 2) /
                   (m * (iterator->x - prev_iterator->x))) -
                  2 * (iterator->z - prev_iterator->z);
      set_R.insert(elemR(current_R, iterator->x, iterator->z, prev_iterator->x,
                         prev_iterator->z));
      iterator++;
      prev_iterator++;
    }

    iteration++;
    auto iterator_R = set_R.begin();
    new_x = (0.5) * (iterator_R->x + iterator_R->x_prev) -
            ((iterator_R->z - iterator_R->z_prev) / (2 * m));
    result = calcOneDim(a2, b2, new_x, func, eps_one, max_iterations_one);
    set.insert(elemTwoDim(result.x, result.y, result.z));

    if (result.z < global_result.z) {
      global_result = result;
    }
    if (iterator_R->x - iterator_R->x_prev <= eps) {
      flag = true;
    }
  }
  return global_result;
}

bool isCorrect(threeDimRes a, threeDimRes b, double eps) {
  bool flag = false;
  if (std::abs(static_cast<double>(a.x - b.x)) <= eps) {
    if (std::abs(static_cast<double>(a.y - b.y)) <= eps) {
      if (std::abs(static_cast<double>(a.z - b.z)) <= eps) {
        flag = true;
      }
    }
  }
  return flag;
}
