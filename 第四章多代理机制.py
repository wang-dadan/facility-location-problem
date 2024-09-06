import numpy as np
import random
import time
import sys
# 定义 h(y, x_i) 函数
def h(y, x_i):
    if x_i <= 0.5:
        if y <= 2 * x_i:
            return 1 - 2 * abs(y - x_i)
        else:
            return 1 - y
    else:
        if y >= 2 * x_i - 1:
            return 1 - 2 * abs(y - x_i)
        else:
            return y
def med(x):
    n = len(x)
    x_sorted = np.sort(x)
    if n % 2 == 0:
        return x_sorted[(n // 2)-1]
    else:
        return x_sorted[(n) // 2]
# 计算中位数

# 计算 f(x) 函数
def f_x(med_x):
    sqrt3_minus_1 = np.sqrt(3) - 1
    if 0 <= med_x <= 2 - np.sqrt(3):
        return 2 - np.sqrt(3)
    elif sqrt3_minus_1 <= med_x <= 1:
        return sqrt3_minus_1
    else:
        return med_x
# 设置循环次数
iterations = 1000
n =6  # 你可以根据需要更改 n 的值

# 存储每次循环的结果
results = []
#中位数机制的近似比
ratio_set_med=[]
#f（x）的近似比
ratio_set_new=[]
alg_runtime_med=[]
alg_runtime_new=[]

for i in range(iterations):
    x = np.random.uniform(0, 1, n)
    x = np.random.uniform(0, 1, n)
    x_rounded = np.round(x, 2)
    # 生成 n 个在 [0, 1] 之间均匀分布的随机数
    x=x_rounded
    x_sorted = np.sort(x)
    x = x_sorted

    # 将 [0,1] 分为 1000 份
    y_values = np.linspace(0, 1, 1000)

    # 计算 SH(y, x) = \sum_{i \in N} h(y, x_i)
    SH_values = np.array([sum(h(y, xi) for xi in x) for y in y_values])

    # 找到使得 SH(y, x) 最大的 y
    y_max_index = np.argmax(SH_values)
    y_max = y_values[y_max_index]
    SH_max = SH_values[y_max_index]
    start_time = time.time()
    # 计算中位数
    med_x = med(x)



    # 计算每个点到中位数的满意度，并按从大到小排序
    satisfactions_med = [h(med_x, xi) for xi in x]
    satisfactions_sorted_med = sorted(satisfactions_med, reverse=True)

    # 计算中位数满意度的总和
    total_satisfaction_med = sum(satisfactions_sorted_med)
    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    alg_runtime_med.append(custom_algorithm_time)

    start_time = time.time()
    # 计算 f(x)
    med_x = med(x)
    f_x_value = f_x(med_x)

    # 计算所有点到 f(x) 的满意度
    satisfactions_f_x = [h(f_x_value, xi) for xi in x]
    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    alg_runtime_new.append(custom_algorithm_time)

    # 计算满意度的总和
    total_satisfaction_f_x = sum(satisfactions_f_x)
    ratio_set_med.append(SH_max/total_satisfaction_med)
    ratio_set_new.append(SH_max/total_satisfaction_f_x)
print(f"  中位数机制最大近似比 是: {max(ratio_set_med)}")
print(f"  中位数机制平均近似比 是: {np.mean(ratio_set_med)}")
print(f"  中位数机制平均运行时间 是: {np.mean(alg_runtime_med)}")

print(f"  机制4.3最大近似比 是: {max(ratio_set_new)}")
print(f"  机制4.3平均近似比 是: {np.mean(ratio_set_new)}")
print(f"  机制4.3平均运行时间 是: {np.mean(alg_runtime_new)}")

print(x)