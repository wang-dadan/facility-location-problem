import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_facility_customer_data(num_facilities, num_customers, max_distance=100, max_opening_cost=50,
                                    max_threshold=80):
    """
    生成设施与客户之间的距离矩阵、设施的开设费用以及距离阈值L。

    参数:
    num_facilities (int): 设施的数量
    num_customers (int): 客户的数量
    max_distance (int): 随机生成距离的最大值
    max_opening_cost (int): 随机生成设施开设费用的最大值
    max_threshold (int): 随机生成距离阈值L的最大值

    返回:
    tuple: (设施与客户之间的距离矩阵, 设施的开设费用列表, 距离阈值L)
    """

    # 生成随机的设施与客户之间的距离矩阵
    D_fc = np.random.randint(10, max_distance, size=(num_facilities, num_customers))

    # 保证对称性 (将客户到设施的距离设置为相同)
    D_cf = D_fc.T

    # 创建完整的距离矩阵 (包括设施到客户和客户到设施的部分)
    D = np.block([[np.zeros((num_facilities, num_facilities)), D_fc],  # 上半部分 (设施-设施 和 设施-客户)
                  [D_cf, np.zeros((num_customers, num_customers))]])  # 下半部分 (客户-设施 和 客户-客户)

    # 强制满足三角不等式
    for i in range(num_facilities):
        for j in range(num_customers):
            for k in range(num_facilities):
                D[i, j + num_facilities] = min(D[i, j + num_facilities],
                                               D[i, k + num_facilities] + D[k, j + num_facilities])

    # 提取设施与客户之间的距离矩阵
    distance_matrix = D[:num_facilities, num_facilities:]

    # 随机生成每个设施的开设费用
    opening_costs = np.random.randint(0, max_opening_cost, size=num_facilities)

    # 随机生成距离阈值L
    L = np.random.randint(1, max_threshold)

    return distance_matrix, opening_costs, L


def calculate_Nj(distance_matrix, L):
    """
    计算每个客户 j 的设施集合 N(j)，即满足 d_ij <= L 的所有设施 i 的集合。

    参数:
    distance_matrix (numpy.ndarray): 设施与客户之间的距离矩阵
    L (int): 距离阈值

    返回:
    dict: 每个客户 j 对应的 N(j) 集合
    """
    num_facilities, num_customers = distance_matrix.shape
    Nj_dict = {}

    for j in range(num_customers):
        Nj = [i for i in range(num_facilities) if distance_matrix[i, j] <= L]
        Nj_dict[j] = Nj

    return Nj_dict


# 设置设施和客户的数量
num_facilities = 100
num_customers = 500

# 生成设施与客户之间的距离矩阵、设施的开设费用以及距离阈值L
facility_customer_distance_matrix, facility_opening_costs, distance_threshold_L = generate_facility_customer_data(
    num_facilities, num_customers)

# 计算每个客户的 N(j)
Nj_dict = calculate_Nj(facility_customer_distance_matrix, distance_threshold_L)

# 输出生成的矩阵、开设费用、距离阈值，以及每个客户的 N(j)
print("设施与客户之间的距离矩阵：")
print(facility_customer_distance_matrix)
print("\n每个设施的开设费用：")
print(facility_opening_costs)
print("\n距离阈值 L：")
print(distance_threshold_L)
print("\n每个客户的 N(j)：")
for j, Nj in Nj_dict.items():
    print(f"客户 {j} 的 N(j)：{Nj}")



def solve_facility_location_with_check(facility_customer_distance_matrix, facility_opening_costs, L):
    num_facilities, num_customers = facility_customer_distance_matrix.shape

    # 计算每个客户的 N(j) 集合
    N = {j: [i for i in range(num_facilities) if facility_customer_distance_matrix[i, j] <= L] for j in range(num_customers)}

    # 检查是否存在空的 N(j)
    for j in range(num_customers):
        if not N[j]:  # 如果 N(j) 是空集
            print(f"客户 {j} 的 N(j) 是空集，无解。")
            return None

    # 如果所有 N(j) 非空，则继续求解
    # 创建模型
    model = gp.Model("facility_location")

    # 添加决策变量
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")  # y_i: 设施 i 是否被开设
    x = model.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name="x")  # x_ij: 设施 i 是否服务客户 j

    # 设置目标函数：最小化设施的开设费用
    model.setObjective(gp.quicksum(facility_opening_costs[i] * y[i] for i in range(num_facilities)), GRB.MINIMIZE)

    # 添加约束条件
    # 约束1: 每个客户 j 必须由一个设施 i 服务
    model.addConstrs((gp.quicksum(x[i, j] for i in range(num_facilities)) == 1 for j in range(num_customers)), "assign")

    # 约束2: 只有开设的设施才能服务客户
    model.addConstrs((x[i, j] <= y[i] for i in range(num_facilities) for j in range(num_customers)), "serve")

    # 约束3: 每个客户 j 到所服务的设施的距离不能超过 L
    model.addConstrs((gp.quicksum(facility_customer_distance_matrix[i, j] * x[i, j] for i in range(num_facilities)) <= L
                      for j in range(num_customers)), "distance")

    # 求解模型
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        print("最优目标值:", model.objVal)
        print("开设的设施:")
        for i in range(num_facilities):
            if y[i].x > 0.5:  # y_i = 1 表示设施 i 被开设
                print(f"设施 {i} 被开设")
        #print("客户分配:")
        # for j in range(num_customers):
        #     for i in range(num_facilities):
        #         if x[i, j].x > 0.5:  # x_ij = 1 表示客户 j 由设施 i 服务
        #             print(f"客户 {j} 由设施 {i} 服务")
    else:
        print("未找到最优解")

# 示例：调用上面函数
solve_facility_location_with_check(facility_customer_distance_matrix, facility_opening_costs, distance_threshold_L)
