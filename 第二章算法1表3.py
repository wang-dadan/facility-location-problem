import numpy as np
import random
import time
import sys
import gurobipy as gp
from gurobipy import GRB

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
    distance = enforce_triangle_inequality(distance_f_c)
    facility_costs = np.random.randint(10, 500, size=num_facilities)
    L = np.random.randint(30, 60)
    return distance, facility_costs, L

# 2. 求解最优解
def solve_optimal_solution(F, D, distances, facility_costs, L):
    model = gp.Model()

    x = model.addVars(F, D, vtype=GRB.BINARY, name="x")
    y = model.addVars(F, vtype=GRB.BINARY, name="y")

    # 目标函数
    model.setObjective(gp.quicksum(facility_costs[i] * y[i] for i in F), GRB.MINIMIZE)

    # 约束条件
    model.addConstrs(gp.quicksum(x[i, j] for i in F) == 1 for j in D)
    model.addConstrs(x[i, j] <= y[i] for i in F for j in D)
    model.addConstrs(gp.quicksum(distances[i, j] * x[i, j] for i in F) <= L for j in D)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', x)
        open_facilities = {i for i in F if y[i].x > 0.5}

        # 修改：生成客户与分配设施的距离矩阵
        assignment_distances = []
        for j in D:
            for i in open_facilities:
                if solution[i, j] > 0.5:
                    assignment_distances.append(distances[i, j])

        assignment_distances = np.array(assignment_distances)
        total_cost = sum(facility_costs[i] for i in open_facilities)
        return total_cost, assignment_distances
    else:
        return None, None


# 3. 自定义算法
def run_custom_algorithm(F, D, distances, facility_costs, L):
    # 计算每个客户j对应的设施集合N(j)
    N_j = {j: [i for i in F if distances[i, j] <= L] for j in D}

    # 检查是否存在N(j)为空的情况
    for j, N in N_j.items():
        if not N:
            print(f"无解，客户 {j} 没有可用的设施.")
            return None, None, None

    # 初始化
    S = set(D)
    open_facilities = set()  # 开设的设施集合
    customer_assignment = {}  # 记录每个客户分配的设施

    # 记录每个客户最小开设费用的设施
    p_j = {j: min(facility_costs[i] for i in N_j[j]) for j in D}
    i_j = {j: min(N_j[j], key=lambda i: facility_costs[i]) for j in D}

    # 记录客户与设施的分配距离矩阵
    assignment_distances = np.zeros(len(D))

    while S:
        # 找到p_j最小的客户j'
        j_prime = min(S, key=lambda j: p_j[j])

        # 关联集群C_j' 和 设施i_j'
        C_j_prime = {j for j in S if not set(N_j[j]).isdisjoint(N_j[j_prime])}
        i_j_prime = i_j[j_prime]

        # 开设设施i_j'
        open_facilities.add(i_j_prime)

        # 为集群C_j'中的每个客户分配设施i_j'
        for j in C_j_prime:
            customer_assignment[j] = i_j_prime
            assignment_distances[j] = distances[i_j_prime, j]

        # 更新S集合，去除已处理的客户
        S -= C_j_prime

    # 计算设施建设费用之和
    total_cost = sum(facility_costs[i] for i in open_facilities)

    # 计算客户分配后的距离矩阵，并求其最大值
    max_distance = np.max(assignment_distances)

    return total_cost, assignment_distances, max_distance


# 主程序
num_facilities = 20
num_customers = 50
num_iterations = 1000

# 记录每次实验的结果
optimal_results = []
custom_results = []
objective_ratio=[]
distance_ratio=[]
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
    if success_count>999:
        break
    # 生成数据
    distances, facility_costs, L = generate_data(num_facilities, num_customers)

    # 输出L的值
    print(f"距离阈值L的值: {L}")

    # 解决最优解
    F = list(range(num_facilities))
    D = list(range(num_customers))
    total_cost_optimal, assignment_distances_optimal = solve_optimal_solution(F, D, distances,facility_costs, L)
    if total_cost_optimal is not None:
        optimal_results.append((total_cost_optimal))
        success_count+=1

    # 运行自定义算法
    start_time = time.time()
    total_cost_custom, assignment_distances_custom, max_distance_custom = run_custom_algorithm(F, D, distances,facility_costs, L)
    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    alg_runtime.append(custom_algorithm_time)

    if total_cost_custom is not None:
        print(f"目标值近似比: {total_cost_custom / total_cost_optimal}")
        print(f"违反约束比例: {max_distance_custom / L}")
        custom_results.append((total_cost_custom, max_distance_custom))
        objective_ratio.append(total_cost_custom / total_cost_optimal)
        distance_ratio.append(max_distance_custom / L)
    if check_triangle_inequality(distances)==True:
        success_intances_count+=1


print(f"最大距离违反比例 alpha_max为: {max(distance_ratio)}")
print(f"最大目标函数比例为beta_max: {max(objective_ratio)}")

print(f"平均距离违反比例为alpha_mean: {np.mean(distance_ratio)}")
print(f"平均目标函数比例为beta_mean: {np.mean(objective_ratio)}")

# print(f"算法运行最大时间: {max(alg_runtime)}")
print(f"算法运行平均时间为t: {np.mean(alg_runtime)}")
print(f"成功实例的个数: {success_count}")
print(f"满足三角不等式实例个数: {success_intances_count}")
print(f"Python 版本: {sys.version}")
# 输出 Gurobi 版本
print(f"Gurobi 版本: {gp.gurobi.version()}")
