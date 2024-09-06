import numpy as np
import random
import time
import sys
import gurobipy as gp
import networkx as nx
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
    distance_matrix = enforce_triangle_inequality(distance_matrix)
    # 下面生成设施与客户之间的距离
    distance_f_c = distance_matrix[:num_facilities, num_facilities:]
    distances = enforce_triangle_inequality(distance_f_c)
    facility_costs = np.random.randint(10, 200, size=num_facilities)
    K = 400
    return distances, facility_costs, K



# 3. 自定义算法
def run_custom_algorithm(F, D, distances, facility_costs, L):
    # 计算每个客户j对应的设施集合N(j)
    N_j = {j: [i for i in F if distances[i, j] <= L] for j in D}

    # 检查是否存在N(j)为空的情况
    for j, N in N_j.items():
        if not N:
            # print(f"无解，客户 {j} 没有可用的设施.")
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
    facility_subset=list(open_facilities)
    # 计算客户分配后的距离矩阵，并求其最大值
    #第一种maxdistance定义方式
    #max_distance = np.max(assignment_distances)

    distances_subset = distances[facility_subset, :]

    # 对于每个客户，计算到设施子集中所有设施的最小距离
    # distances_subset[i, :] 是第 i 个设施子集的所有客户的距离
    min_distances = np.min(distances_subset, axis=0)

    # 计算所有客户的最大最小距离
    max_distance = np.max(min_distances)

    return total_cost, assignment_distances, max_distance

def algorithm_2_2(F, D, distances, facility_costs, K):
    # 步骤 1：构造 C 集合
    C = sorted(set(distances[i][j] for i in F for j in D))

    while C:
        # 步骤 2：选择 C 中的最小值
        c_bar = C[0]

        # 使用 solve_optimal_solution 函数对 P_c_bar 进行求解
        total_cost, assignment_distances,max_distance = run_custom_algorithm(F, D, distances, facility_costs, c_bar)

        if total_cost is not None:
            if max_distance <= 3 * c_bar and total_cost <= k:
                return max_distance # 输出解
        # else:
        #     print(f"Optimal solution not found for c_bar = {c_bar}")

        # 如果不满足条件，移除 c_bar 并回到步骤 2
        C = C[1:]

    # 如果循环完所有 C 依然没有找到满足条件的解，返回 None
    return None, None


def construct_G_from_distances(distances, facility_costs, k):
    num_facilities, num_customers = distances.shape
    G = nx.Graph()

    # Adding edges between facilities and customers
    for i in range(num_facilities):
        for j in range(num_customers):
            G.add_edge(i, num_facilities + j, weight=distances[i, j])

    return G


def algorithm2(G, k):
    def construct_G_i(G, c_e_i):
        G_i = nx.Graph()
        for u, v, data in G.edges(data=True):
            if data['weight'] <= c_e_i:
                G_i.add_edge(u, v, weight=data['weight'])
        return G_i

    def construct_G_i2(G_i):
        G_i2 = nx.Graph()
        for u in G_i.nodes:
            for v in G_i.nodes:
                if u != v:
                    if nx.has_path(G_i, u, v) and len(nx.shortest_path(G_i, u, v)) <= 2:
                        G_i2.add_edge(u, v)
        return G_i2

    def find_maximal_independent_set(G):
        independent_set = set()
        nodes = list(G.nodes)
        while nodes:
            node = nodes.pop(0)
            independent_set.add(node)
            neighbors = list(G.neighbors(node))
            nodes = [n for n in nodes if n not in neighbors]
        return independent_set

    # Extract edge lengths from graph G
    edge_lengths = sorted(set(data['weight'] for u, v, data in G.edges(data=True)))

    # Define weights directly from the facility_costs array
    weights = {i: cost for i, cost in enumerate(facility_costs)}

    for c_e_i in edge_lengths:
        G_i = construct_G_i(G, c_e_i)
        all_customers_connected = True
        for customer in range(num_facilities, num_facilities + num_customers):
            if not any(G_i.has_edge(customer, supplier) for supplier in range(num_facilities)):
                all_customers_connected = False
                break

        if not all_customers_connected:
            continue

        G_i2 = construct_G_i2(G_i)
        G_i2_D = G_i2.subgraph(range(num_facilities, num_facilities + num_customers))

        S = find_maximal_independent_set(G_i2_D)

        S_prime = set()
        for customer in S:
            neighbors = list(G_i.neighbors(customer))
            valid_neighbors = [n for n in neighbors if n in weights]
            if valid_neighbors:
                min_weight_supplier = min(valid_neighbors, key=lambda x: weights[x])
                S_prime.add(min_weight_supplier)

        if sum(weights.get(supplier, 0) for supplier in S_prime) <= k:
            return c_e_i

    return None


# 主程序
num_facilities = 10
num_customers = 30
num_iterations = 1000
# Define k (upper bound on total weight of selected facilities)
k = 400

# 记录每次实验的结果
our_optimal_results = []
our_custom_results = []
our_objective_ratio=[]
our_alg_runtime=[]
success_intances_count=0
old_alg_runtime=[]
old_objective_ratio=[]
old_result=[]

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
    distances, facility_costs, K = generate_data(num_facilities, num_customers)


    # 解决最优解
    F = list(range(num_facilities))
    D = list(range(num_customers))

    model = gp.Model()

    x = model.addVars(F, D, vtype=GRB.BINARY, name="x")
    y = model.addVars(F, vtype=GRB.BINARY, name="y")
    # 引入新的变量 z 表示最大距离
    z = model.addVar(name="z")

    # 目标函数
    model.setObjective(z, GRB.MINIMIZE)

    # 约束条件
    model.addConstrs(gp.quicksum(x[i, j] for i in F) == 1 for j in D)
    model.addConstrs(x[i, j] <= y[i] for i in F for j in D)
    model.addConstr(gp.quicksum(facility_costs[i] * y[i] for i in F) <= K)
    # 添加约束：z >= sum(distances[i, j] * x[i, j] for i in F) 对于每个 j ∈ D
    model.addConstrs((gp.quicksum(distances[i, j] * x[i, j] for i in F) <= z for j in D), name="max_distance")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', x)
        open_facilities = {i for i in F if y[i].x > 0.5}

        # 修改：生成客户与分配设施的距离矩阵
        assignment_distances = []
        optimal_obj_value = model.objVal
        for j in D:
            for i in open_facilities:
                if solution[i, j] > 0.5:
                    assignment_distances.append(distances[i, j])

        assignment_distances = np.array(assignment_distances)
        total_cost_opt = sum(facility_costs[i] for i in open_facilities)
        z = model.getVarByName('z')
        z_value = z.x if z else None
        print(f"最优目标函数值: {optimal_obj_value}")
        our_optimal_results.append(optimal_obj_value)
        if z_value is not None:
            print(f"变量 z 的值: {z_value}")
        else:
            print("变量 z 不存在")
    # return total_cost, assignment_distances
    # 运行自定义算法
    start_time = time.time()
    result = algorithm_2_2(F, D, distances, facility_costs, K)
    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    our_alg_runtime.append(custom_algorithm_time)
    if result:
        print("算法解已得到:")
        print("算法解为:", result)
        our_custom_results.append(result)
        our_objective_ratio.append(result/optimal_obj_value)
        # print("Assignment distances:", assignment_distances)
    else:
        print("No solution satisfies the conditions.")
    if check_triangle_inequality(distances)==True:
        success_intances_count+=1
    start_time = time.time()
    G = construct_G_from_distances(distances, facility_costs, K)
    shmoys_result = algorithm2(G, k)
    end_time = time.time()
    print(f"Start Time: {start_time}, End Time: {end_time}")
    old_algorithm_time = end_time - start_time
    old_alg_runtime.append(old_algorithm_time)
    old_result.append(shmoys_result)
    old_objective_ratio.append(shmoys_result/optimal_obj_value)


print(f"our最坏近似比 为: {max(our_objective_ratio)}")
print(f"our平均近似比为: {np.mean(our_objective_ratio)}")

# print(f"算法运行最大时间: {max(alg_runtime)}")
print(f"our算法运行平均时间为t: {np.mean(our_alg_runtime)}")
print(f"shmoys算法最坏近似比 为: {max(old_objective_ratio)}")
print(f"shmoys算法平均近似比为: {np.mean(old_objective_ratio)}")

# print(f"算法运行最大时间: {max(alg_runtime)}")
print(f"shmoys算法运行平均时间为t: {np.mean(old_alg_runtime)}")
print(f"满足三角不等式实例个数: {success_intances_count}")



print(f"Python 版本: {sys.version}")

# 输出 Gurobi 版本
print(f"Gurobi 版本: {gp.gurobi.version()}")