import numpy as np

max_distance = 100  # 可以调整最大距离的值
num_facilities = 20  # 示例设施数量
num_customers = 50  # 示例客户数量

def generate_symmetric_matrix(size):
    matrix = np.random.randint(1, max_distance, size=(size, size))
    matrix = (matrix + matrix.T) / 2  # 确保对称
    np.fill_diagonal(matrix, 0)  # 对角线为零
    return matrix

def generate_fc_matrix(num_facilities, num_customers):
    return np.random.randint(1, max_distance, size=(num_facilities, num_customers))

def enforce_triangle_inequality(matrix):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            for k in range(m):  # k 的范围应在行数范围内
                for t in range(n):
                    if matrix[i, j] > matrix[i, t] + matrix[k, t] + matrix[k, j]:
                        matrix[i, j] = matrix[i, t] + matrix[k, t] + matrix[k, j]
    return matrix

# 生成设施与客户之间的距离矩阵
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
#下面生成设施与客户之间的距离
distance_f_c = distance_matrix[:num_facilities, num_facilities:]
distance_f_c = enforce_triangle_inequality(distance_f_c)

def check_triangle_inequality(matrix):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            for k in range(m):  # k 的范围应在行数范围内
                for t in range(n):
                    if matrix[i, j] > matrix[i, t] + matrix[k, t]+matrix[k, j]:
                        return False
    return True

print("Final distance matrix:\n", distance_f_c)

print(distance_f_c)
rows, cols = distance_f_c.shape
print(rows,cols)
print("Does the distance matrix satisfy the triangle inequality?", check_triangle_inequality(distance_f_c))