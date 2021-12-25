// Copyright 2021 Troegubova Alexandra
#ifndef MODULES_TASK_3_TROEGUBOVA_A_GRAHAM_SCAN_GRAHAM_SCAN_H_
#define MODULES_TASK_3_TROEGUBOVA_A_GRAHAM_SCAN_GRAHAM_SCAN_H_

#include <vector>
#include<stack>

std::vector<int> GetPoints(int count_p, int centre_x, int centre_y, int r);
int Rotate(const std::vector<int>& p, const std::vector<int>& b, const std::vector<int>& c);
int Distance(const std::vector<int>& p1, const std::vector<int>& p2);
int Compare(const void* p_1, const void* p_2);
std::vector<int> NextToTop(std::stack<std::vector<int>> S);
std::vector<int> GrahamSequential(const std::vector<int>& p, int count_points, int* size_shell);
std::vector<int> GrahamParallel(const std::vector<int>& points, int count_points, int* size_shell);

#endif  // MODULES_TASK_3_TROEGUBOVA_A_GRAHAM_SCAN_GRAHAM_SCAN_H_





