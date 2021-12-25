// Copyright 2021 Labinskaya Marina
#include <random>
#include <cmath>
#include "../../../modules/task_3/labinskaya_m_mult_matr_cannon_algorithm/cannon_algorithm.h"

double* creating_zero_matrix(int size) {
    double* matr = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        matr[i] = 0;
    }
    return matr;
}

double* creating_random_matrix(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);
    double* matr = new double[size * size];
    for (int i = 0; i < size * size; i++) {
        matr[i] = dis(gen);
    }
    return matr;
}

double* sequantial_mult(int size, double* matr1, double* matr2) {
    double* res = creating_zero_matrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int l = 0; l < size; l++) {
                res[i * size + j] += matr1[i * size + l] * matr2[l * size + j];
            }
        }
    }
    return res;
}

MPI_Comm creating_decart_topology(int num) {
    int ProcNum;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (num > 0 && (num * num <= ProcNum)) {
        int dims[2] = { num, num };
        int periods[2] = { 1, 1 };
        MPI_Comm decart_comm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &decart_comm);
        return decart_comm;
    } else {
        return MPI_COMM_NULL;
    }
}

double* parallel_mult(int size, double* matr1, double* matr2) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if ((static_cast<int>(std::sqrt(ProcNum)) * static_cast<int>(std::sqrt(ProcNum)) != ProcNum) ||
        (size % static_cast<int>(sqrt(ProcNum)) != 0)) {
        if (ProcRank == 0) {
            return sequantial_mult(size, matr1, matr2);
        } else {
            return creating_zero_matrix(size);
        }
    }

    double* res = creating_zero_matrix(size);

    int block_num = static_cast<int>(std::sqrt(ProcNum));
    int block_size = size / block_num;

    int matrix_sizes[2] = { size, size };
    int block_sizes[2] = { block_size, block_size };
    int starts[2] = { 0, 0 };

    MPI_Datatype type, new_type;
    MPI_Type_create_subarray(2, matrix_sizes, block_sizes, starts, MPI_ORDER_C,
    MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, block_size * sizeof(double), &new_type);
    MPI_Type_commit(&new_type);

    int* sendcounts = new int[ProcNum];
    int* displs = new int[ProcNum];

    if (ProcRank == 0) {
        for (int i = 0; i < ProcNum; i++) {
            sendcounts[i] = 1;
        }
        int temp_displs = 0;
        for (int i = 0; i < block_num; i++) {
            for (int j = 0; j < block_num; j++) {
                displs[i * block_num + j] = temp_displs;
                temp_displs += 1;
            }
            temp_displs += (block_size - 1) * block_num;
        }
    }

    MPI_Comm decart_comm = creating_decart_topology(block_num);

    double* local_matr1 = new double[block_size * block_size];
    double* local_matr2 = new double[block_size * block_size];

    MPI_Scatterv(matr1, sendcounts, displs, new_type, local_matr1,
    (size * size) / ProcNum, MPI_DOUBLE, 0, decart_comm);
    MPI_Scatterv(matr2, sendcounts, displs, new_type, local_matr2,
    (size * size) / ProcNum, MPI_DOUBLE, 0, decart_comm);

    int coords[2];
    MPI_Cart_coords(decart_comm, ProcRank, 2, coords);

    MPI_Status status;
    int left, right, up, down;
    MPI_Cart_shift(decart_comm, 1, coords[0], &left, &right);
    MPI_Sendrecv_replace(local_matr1, block_size * block_size, MPI_DOUBLE, left, 1,
    right, 1, decart_comm, &status);

    MPI_Cart_shift(decart_comm, 0, coords[1], &up, &down);
    MPI_Sendrecv_replace(local_matr2, block_size * block_size, MPI_DOUBLE, up,
    1, down, 1, decart_comm, &status);

    double* local_res = creating_zero_matrix(block_size);
    double* temp_res = creating_zero_matrix(block_size);

    for (int q = 0; q < block_num; q++) {
    temp_res = sequantial_mult(block_size, local_matr1, local_matr2);
        for (int i = 0; i < block_size * block_size; i++) {
            local_res[i] += temp_res[i];
        }

        MPI_Cart_shift(decart_comm, 1, 1, &left, &right);
        MPI_Cart_shift(decart_comm, 0, 1, &up, &down);

        MPI_Sendrecv_replace(local_matr1, block_size * block_size, MPI_DOUBLE, left,
        1, right, 1, decart_comm, &status);
        MPI_Sendrecv_replace(local_matr2, block_size * block_size, MPI_DOUBLE, up,
        1, down, 1, decart_comm, &status);
    }
    MPI_Gatherv(local_res, size * size / ProcNum, MPI_DOUBLE, res,
    sendcounts, displs, new_type, 0, decart_comm);
    return res;
}
