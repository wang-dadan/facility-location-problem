import numpy as np

# 给定的距离矩阵
distance_matrix = np.array([265, 359, 324, 59, 219]
[272, 288, 375, 176, 49]
[256, 419, 338, 50, 210]
[384, 556, 392, 286, 316]
[152, 345, 253, 54, 106]
)

def check_triangle_inequality(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if matrix[i, j] > matrix[i, k] + matrix[k, j]:
                    print(f"Triangle inequality violated for i={i}, j={j}, k={k}:")
                    print(f"D(i, j) = {matrix[i, j]}")
                    print(f"D(i, k) + D(k, j) = {matrix[i, k]} + {matrix[k, j]}")
                    return False
    return True

# 验证是否满足三角不等式
is_valid = check_triangle_inequality(distance_matrix)

if is_valid:
    print("The distance matrix satisfies the triangle inequality.")
    print("满足")
else:
    print("The distance matrix does not satisfy the triangle inequality.")
