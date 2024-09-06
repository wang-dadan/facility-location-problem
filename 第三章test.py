import numpy as np
import random
import time
import sys
import gurobipy as gp
from gurobipy import GRB


# 1. 随机生成数据

import numpy as np

def generate_data(num_facilities, num_customers):
    max_distance = 100  # 可以调整最大距离的值

    def generate_symmetric_matrix(size):
        matrix = np.random.randint(10, max_distance, size=(size, size))
        matrix = (matrix + matrix.T) / 2  # 确保对称
        np.fill_diagonal(matrix, 0)  # 对角线为零
        return matrix

    def generate_fc_matrix(num_facilities, num_customers):
        return np.random.randint(10, max_distance, size=(num_facilities, num_customers))

    def enforce_triangle_inequality(matrix):
        m, n = matrix.shape
        for i in range(m):
            for j in range(n):
                for k in range(m):  # k 的范围应在行数范围内
                    for t in range(n):
                        if matrix[i, j] > matrix[i, t] + matrix[k, t] + matrix[k, j]:
                            matrix[i, j] = matrix[i, t] + matrix[k, t] + matrix[k, j]
        return matrix

    def check_triangle_inequality(matrix):
        m, n = matrix.shape
        for i in range(m):
            for j in range(n):
                for k in range(m):  # k 的范围应在行数范围内
                    for t in range(n):
                        if matrix[i, j] > matrix[i, t] + matrix[k, t] + matrix[k, j]:
                            return False
        return True

    # 生成距离矩阵
    D_ff = generate_symmetric_matrix(num_facilities)
    D_cc = generate_symmetric_matrix(num_customers)
    D_fc = generate_fc_matrix(num_facilities, num_customers)

    # 初始化完整的距离矩阵
    distance_matrix = np.zeros((num_facilities + num_customers, num_facilities + num_customers))

    # 填充设施之间的距离
    distance_matrix[:num_facilities, :num_facilities] = D_ff

    # 填充客户之间的距离
    distance_matrix[num_facilities:, num_facilities:] = D_cc

    # 填充设施和客户之间的距离
    distance_matrix[:num_facilities, num_facilities:] = D_fc
    distance_matrix[num_facilities:, :num_facilities] = D_fc.T

    # 应用三角不等式修正
    # 应用三角不等式修正
    distance_matrix = enforce_triangle_inequality(distance_matrix)
    # 下面生成设施与客户之间的距离
    distance_f_c = distance_matrix[:num_facilities, num_facilities:]
    distance_f_c = enforce_triangle_inequality(distance_f_c)

    # 提取设施与客户之间的距离矩阵
    penalty_costs = np.random.randint(20, 200, size=num_customers)
    K = 2  # 确保在后续代码中有解释其用途
    return distance_f_c, penalty_costs, K

def solve_facility_location(distances, penalty_costs, K):
    # 创建模型
    model = gp.Model("facility_location")

    # 决策变量
    x = model.addVars(num_facilities, num_customers, K, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_facilities, K, vtype=GRB.BINARY, name="y")
    z = model.addVars(num_customers, vtype=GRB.BINARY, name="z")

    # 目标函数
    model.setObjective(
        gp.quicksum(distances[i, j] * x[i, j, l] for i in range(num_facilities) for j in range(num_customers) for l in range(K)) +
        gp.quicksum(penalty_costs[j] * z[j] for j in range(num_customers)),
        GRB.MINIMIZE
    )

    # 约束条件
    for j in range(num_customers):
        for l in range(K):
            model.addConstr(gp.quicksum(x[i, j, l] for i in range(num_facilities)) + z[j] == 1)

    for i in range(num_facilities):
        for l in range(K):
            for j in range(num_customers):
                model.addConstr(y[i, l] >= x[i, j, l])

    for i in range(num_facilities):
        model.addConstr(gp.quicksum(y[i, l] for l in range(K)) == 1)

    # 求解模型
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        solution_x = model.getAttr('x', x)
        solution_y = model.getAttr('x', y)
        solution_z = model.getAttr('x', z)
        print(f"Optimal objective value: {model.ObjVal}")
        result=model.ObjVal
        return result
    else:
        print("No optimal solution found.")
        return None

def calculate_alpha_sum(distances, penalty_costs, k):
    alpha_sum = 0
    for j in range(num_customers):
        # 找到与客户 j 最近的 k 个设施的距离之和
        customer_distances = distances[:, j]
        closest_distances = np.sort(customer_distances)[:k]
        c_bar_j = np.sum(closest_distances)

        # 计算 alpha_j = min(c_bar_j, p_j)
        alpha_j = min(c_bar_j, penalty_costs[j])

        # 将 alpha_j 加到总和中
        alpha_sum += alpha_j

    return alpha_sum


# 主程序
num_facilities = 20
num_customers = 50
num_iterations = 1000

# 记录每次实验的结果
optimal_results = []
costsharing_results = []
objective_ratio=[]
success_count=0
alg_runtime=[]
success_intances_count=0

def check_triangle_inequality(matrix):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            for k in range(m):  # k 的范围应在行数范围内
                for t in range(n):
                    if matrix[i, j] > matrix[i, t] + matrix[k, t]+matrix[k, j]:
                        return False
    return True


for _ in range(num_iterations):
    # 生成数据
    distances, penalty_costs, K = generate_data(num_facilities, num_customers)
    # 解决最优解
    F = list(range(num_facilities))
    D = list(range(num_customers))
    result = solve_facility_location(distances, penalty_costs, K)
    # 运行自定义算法
    start_time = time.time()
    # 求解分摊总费用
    alpha_sum = calculate_alpha_sum(distances, penalty_costs, K)
    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    alg_runtime.append(custom_algorithm_time)
    if result is not None:
        optimal_results.append(result)
    objective_ratio.append(result/alpha_sum)
    costsharing_results.append(alpha_sum)
    if check_triangle_inequality(distances)==True:
        success_intances_count+=1



rows, cols = distances.shape
print(rows,cols)
print(f"最坏近似比 为: {max(objective_ratio)}")
print(f"平均近似比为: {np.mean(objective_ratio)}")

# print(f"算法运行最大时间: {max(alg_runtime)}")
print(f"算法运行平均时间为t: {np.mean(alg_runtime)}")
print(f"满足三角不等式的实例为: {success_intances_count}")