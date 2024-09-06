import numpy as np
import time


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


# 计算 f(x) 函数
def f(x1, x2):
    if x1 <= x2 <= 0.5:
        return x2
    elif 0.5 < x1 <= x2:
        return x1
    elif x1 < 0.5 < x2:
        return 0.5
    else:
        # 如果 x1 和 x2 都不满足任何条件，返回 None 或其他适当的值
        return None


# 设置循环次数
iterations = 1000
n = 2  # 你可以根据需要更改 n 的值

# 存储每次循环的结果
ratio_set_new = []
alg_runtime_new = []

for i in range(iterations):
    # 生成 n 个在 [0, 1] 之间均匀分布的随机数
    x = np.random.uniform(0, 1, n)
    x_rounded = np.round(x, 2)  # 四舍五入到两位小数
    x_sorted = np.sort(x_rounded)  # 排序
    x = x_sorted

    # 将 [0, 1] 分为 1000 份
    y_values = np.linspace(0, 1, 1000)

    # 计算 SH(y, x) = \sum_{i \in N} h(y, x_i)
    SH_values = np.array([sum(h(y, xi) for xi in x) for y in y_values])

    # 找到使得 SH(y, x) 最大的 y
    y_max_index = np.argmax(SH_values)
    y_max = y_values[y_max_index]
    SH_max = SH_values[y_max_index]

    start_time = time.time()

    # 计算 f(x)
    x__1 = x[0]
    x__2 = x[1]

    f_x_value = f(x__1, x__2)

    # 处理 f_x_value 为 None 的情况
    if f_x_value is None:
        ratio_set_new.append(None)
        alg_runtime_new.append(None)
        continue

    # 计算所有点到 f(x) 的满意度
    satisfactions_f_x = [h(f_x_value, xi) for xi in x]

    end_time = time.time()
    custom_algorithm_time = end_time - start_time
    alg_runtime_new.append(custom_algorithm_time)

    # 计算满意度的总和
    total_satisfaction_f_x = sum(satisfactions_f_x)

    # 防止除以零错误
    if total_satisfaction_f_x == 0:
        ratio_set_new.append(None)
    else:
        ratio_set_new.append(SH_max / total_satisfaction_f_x)

# 过滤掉 None 值
ratio_set_filtered = [r for r in ratio_set_new if r is not None]
alg_runtime_filtered = [t for t in alg_runtime_new if t is not None]

if ratio_set_filtered:  # 确保 ratio_set_filtered 不为空
    print(f"机制4.1最大近似比 是: {np.max(ratio_set_filtered)}")
    print(f"机制4.1平均近似比 是: {np.mean(ratio_set_filtered)}")
else:
    print("没有有效的近似比数据。")

if alg_runtime_filtered:  # 确保 alg_runtime_filtered 不为空
    print(f"机制4.1平均运行时间 是: {np.mean(alg_runtime_filtered)}")
else:
    print("没有有效的运行时间数据。")

print(x)
