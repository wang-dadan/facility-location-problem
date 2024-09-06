import numpy as np

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

# 生成实例数据
np.random.seed(42)
F = range(10)  # 设施集合
D = range(50)  # 客户集合
distances = np.random.randint(1, 20, size=(len(F), len(D)))
facility_costs = np.random.randint(10, 50, size=len(F))
L = np.random.randint(10, 20)  # 随机生成距离阈值

# 运行算法并输出结果
total_cost, assignment_distances, max_distance = run_custom_algorithm(F, D, distances, facility_costs, L)

if total_cost is not None:
    print(f"算法开设的设施建设费用之和: {total_cost}")
    print(f"客户与分配设施的距离矩阵: {assignment_distances}")
    print(f"最大客户与设施距离: {max_distance}")
