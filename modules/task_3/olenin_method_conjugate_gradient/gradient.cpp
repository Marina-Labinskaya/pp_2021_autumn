// Copyright 2021 Olenin Sergey
#include <mpi.h>
#include <ctime>
#include <random>
#include <vector>
#include "../../../modules/task_3/olenin_method_conjugate_gradient/gradient.h"

std::vector<double> rand_matrix(int size) {
  if (0 >= size) throw "wrong matrix size";

  std::default_random_engine gene;
  gene.seed(static_cast<unsigned int>(time(0)));

  int sizematrix = size * size;

  std::vector<double> rand_matrix(sizematrix);

  for (int Index = 0; Index < size; ++Index) {
    for (int Jindex = Index; Jindex < size; ++Jindex) {
      rand_matrix[Index * size + Jindex] = gene() % 10;
      rand_matrix[Jindex * size + Index] = rand_matrix[Index * size + Jindex];
    }
  }

  return rand_matrix;
}

std::vector<double> rand_vec(int size) {
  if (0 >= size) throw "wrong size of vector";

  std::default_random_engine gene;

  gene.seed(static_cast<unsigned int>(time(0)));

  std::vector<double> vec(size);

  for (int index = 0; index < size; ++index) {
    vec[index] = 1 + gene() % 10;
  }

  return vec;
}

double vec_mult(const std::vector<double>& vec_a,
                const std::vector<double>& vec_b) {
  if (vec_a.size() != vec_b.size()) throw "vector sizes are not consistent ";

  double multi = 0.0;

  for (size_t index = 0; index < vec_a.size(); ++index)
    multi += vec_a[index] * vec_b[index];

  return multi;
}

std::vector<double> matrix_vec_mult(const std::vector<double>& matrix,
                                     const std::vector<double>& vec) {
  if ((matrix.size() % vec.size()) != 0)
    throw "matrix and vector sizes are incompatible";

  std::vector<double> multi(matrix.size() / vec.size());

  for (size_t Index = 0; Index < (matrix.size() / vec.size()); ++Index) {
    multi[Index] = 0.0;
    for (size_t Jindex = 0; Jindex < vec.size(); ++Jindex) {
      multi[Index] += matrix[Index * vec.size() + Jindex] * vec[Jindex];
    }
  }

  return multi;
}

std::vector<double> get_sol_for_one_proc(const std::vector<double>& matrix,
                                const std::vector<double>& vec, int size) {
  if (size <= 0) throw "wrong size";

  int iteration = 0;

  double alpha = 0.0, beta = 0.0;

  double norm_residual = 0.0, criteria = 0.1;

  std::vector<double> x(size);

  for (int index = 0; index < size; index++) {
    x[index] = 1;
  }

  std::vector<double> Az = matrix_vec_mult(matrix, x);

  std::vector<double> residualprev(size), residualnext(size);

  for (int index = 0; index < size; index++)
    residualprev[index] = vec[index] - Az[index];
  // residualprev^(k) = b - A * x

  std::vector<double> gradient(residualprev);
  // gradient = r^(k)

  do {
    iteration++;

    Az = matrix_vec_mult(matrix, gradient);
    // Az = A * gradient

    alpha = vec_mult(residualprev, residualprev) / vec_mult(gradient, Az);

    for (int index = 0; index < size; index++) {
      x[index] += alpha * gradient[index];
      residualnext[index] = residualprev[index] - alpha * Az[index];
      // residualnext^(k+1) = residualprev^(k)-alpha*Az
    }

    beta = vec_mult(residualnext, residualnext) /
           vec_mult(residualprev, residualprev);

    norm_residual = sqrt(vec_mult(residualnext, residualnext));

    for (int index = 0; index < size; index++)
      gradient[index] = residualnext[index] + beta * gradient[index];
    // gradient^(k+1) = residualnext^(k+1) + beta*gradient^(k)

    residualprev = residualnext;
  } while ((norm_residual > criteria) && (iteration <= size));

  return x;
}

std::vector<double> get_sol_for_multiply_proc(
    const std::vector<double>& matrix,
                                const std::vector<double>& vec,
                                int size) {
  if (0 >= size) throw "wrong size";

  std::vector<double> matrixtmp = matrix;

  std::vector<double> vectortmp = vec;

  MPI_Bcast(matrixtmp.data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(vectortmp.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int size_comm, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int k_rows = size / size_comm;

  int balance = size % size_comm;

  std::vector<double> matrixL(k_rows * size);

  if (0 == rank) {
    if (0 != balance) {
      matrixL.resize(size * k_rows + balance * size);
    }
    if (0 != k_rows) {
      for (int process = 1; process < size_comm; process++) {
        MPI_Send(&matrixtmp[0] + process * k_rows * size + balance * size,
                 k_rows * size, MPI_DOUBLE, process, 1, MPI_COMM_WORLD);
      }
    }
  }
  if (0 == rank) {
    if (balance != 0) {
      for (int index = 0; index < size * k_rows + size * balance; index++) {
        matrixL[index] = matrixtmp[index];
      }
    } else {
      for (int index = 0; index < size * k_rows; index++) {
        matrixL[index] = matrixtmp[index];
      }
    }
  } else {
    MPI_Status message;

    if (k_rows != 0) {
      MPI_Recv(&matrixL[0], k_rows * size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
               &message);
    }
  }
  int iteration = 0;

  double beta = 0.0, alpha = 0.0;

  double norm_residual = 0.0, criteria = 0.1;

  std::vector<double> x(size);

  for (int index = 0; index < size; index++) {
    x[index] = 1;
  }

  std::vector<double> Az = matrix_vec_mult(matrixL, x);

  std::vector<double> residualprev(k_rows), residualnext(k_rows);

  if (0 == rank) {
    if (0 != balance) {
      residualprev.resize(k_rows + balance);
      residualnext.resize(k_rows + balance);
    }
    for (int index = 0; index < k_rows + balance; index++)
      residualprev[index] = vectortmp[index] - Az[index];
    // residualprev^(k) = b - A * x
  } else {
    for (int index = 0; index < k_rows; index++)
      residualprev[index] =
          vectortmp[rank * k_rows + balance + index] - Az[index];
    // residualprev^(k) = b - A * x
  }

  std::vector<double> gradient(size);
  // gradient = residualprev^(k)
  if (rank == 0) {
    if (balance != 0) {
      for (int index = 0; index < k_rows + balance; index++) {
        gradient[index] = residualprev[index];
      }
    } else {
      for (int index = 0; index < k_rows; index++) {
        gradient[index] = residualprev[index];
      }
    }
    if (k_rows != 0) {
      MPI_Status message;
      for (int process = 1; process < size_comm; process++) {
        MPI_Recv(&gradient[0] + process * k_rows + balance, k_rows, MPI_DOUBLE,
                 process, 2, MPI_COMM_WORLD, &message);
      }
    }
  } else {
    if (k_rows != 0) {
      MPI_Send(&residualprev[0], k_rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
  }

  MPI_Bcast(gradient.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> gradientblock(k_rows);
  std::vector<double> gradientlocal(k_rows);

  if (rank == 0) {
    if (balance != 0) {
      gradientblock.resize(k_rows + balance);
    }
  }
  do {
    iteration++;
    Az = matrix_vec_mult(matrixL, gradient);  // Az = A * gradient

    if (rank == 0) {
      for (int index = 0; index < k_rows + balance; index++) {
        gradientblock[index] = gradient[index];
      }
    } else {
      for (int index = 0; index < k_rows; index++) {
        gradientblock[index] = gradient[rank * k_rows + balance + index];
      }
    }

    double tmp_f = vec_mult(residualprev, residualprev);
    double tmp_s = vec_mult(gradientblock, Az);

    double residualprevscalar;
    double devideralpha;

    MPI_Allreduce(&tmp_f, &residualprevscalar, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(&tmp_s, &devideralpha, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    alpha = residualprevscalar / devideralpha;

    for (int index = 0; index < size; index++) {
      x[index] += alpha * gradient[index];
    }

    if (rank == 0) {
      for (int index = 0; index < k_rows + balance; index++) {
        residualnext[index] = residualprev[index] - alpha * Az[index];
        // residualnext^(k+1) = residualprev^(k)-alpha*Az
      }
    } else {
      for (int index = 0; index < k_rows; index++) {
        residualnext[index] = residualprev[index] - alpha * Az[index];
        // residualnext^(k+1) = residualprev^(k)-alpha*Az
      }
    }


    double residualnextscalar;

    tmp_f = vec_mult(residualnext, residualnext);

    MPI_Allreduce(&tmp_f, &residualnextscalar, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    beta = residualnextscalar / residualprevscalar;

    norm_residual = sqrt(residualnextscalar);

    if (rank == 0) {
      for (int index = 0; index < k_rows + balance; index++) {
        gradient[index] = residualnext[index] + beta * gradient[index];
      }

      if (k_rows != 0) {
        MPI_Status message;
        for (int process = 1; process < size_comm; process++) {
          MPI_Recv(&gradient[0] + process * k_rows + balance,
            k_rows, MPI_DOUBLE, process, 3, MPI_COMM_WORLD, &message);
        }
      }
    } else {
      for (int index = 0; index < k_rows; index++) {
        gradientlocal[index] = residualnext[index] +
                               beta * gradient[rank * k_rows + balance + index];
      }

      if (k_rows != 0) {
        MPI_Send(&gradientlocal[0], k_rows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
      }
    }
    MPI_Bcast(gradient.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    residualprev = residualnext;
  } while ((norm_residual > criteria) && (iteration <= size));
  return x;
}
