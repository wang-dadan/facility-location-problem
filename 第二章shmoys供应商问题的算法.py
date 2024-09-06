import networkx as nx
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
                for k in range(m):
                    for t in range(n):
                        if matrix[i, j] > matrix[i, t] + matrix[k, t] + matrix[k, j]:
                            matrix[i, j] = matrix[i, t] + matrix[k, t] + matrix[k, j]
        return matrix

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
    weight = 400
    return distances, facility_costs, weight


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
            return S_prime

    return None


# Example usage with generated data
num_facilities = 5
num_customers = 4
distances, facility_costs, weight = generate_data(num_facilities, num_customers)
print("Distances matrix shape:", distances.shape)  # Debugging line
G = construct_G_from_distances(distances, facility_costs, weight)
print("Graph nodes:", G.nodes)  # Debugging line
print("Graph edges:", G.edges)  # Debugging line

# Define k (upper bound on total weight of selected facilities)
k = 400

# Run the algorithm
result = algorithm2(G, k)
print("Optimal suppliers:", result)
