import gurobipy as gp
from gurobipy import GRB

import numpy as np
from numpy.ma.core import floor

from Graph_clustering import clustering
from Graph_generation import Graph

# делаем граф
n_vertices = 70
avg_n_edges = 15
n_edges = n_vertices * avg_n_edges
n_colors = 6

rs = np.random.RandomState(42)


# TODO: constrs =n_vertices +n_edges*n_colors
def gurobi_optimisation(graph: Graph, n_colors: int):
    # edges = {}
    # for i in graph.graph:
    #     for j in graph.graph[i]:
    #         if (j,i) not in edges:
    #             edges[i,j]=graph.graph[i][j]
    edges = graph.transform_graph_with_mapping()[0]
    print(len(graph.graph))
    print(len(edges) * n_colors + len(graph.graph))
    print(edges)
    m = gp.Model()
    timeout = 120

    # создаём переменные
    x = np.zeros((n_vertices, n_colors), dtype=object)
    for v in range(n_vertices):
        for c in range(n_colors):
            x[v][c] = m.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
    y = np.zeros((n_edges, n_colors), dtype=object)
    for e, (i, j) in enumerate(edges):
        for c in range(n_colors):
            y[e][c] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{c}")

    # целевая функция
    objective = gp.quicksum([w * y[e][c] \
                             for e, w in enumerate(edges.values()) \
                             for c in range(n_colors)] \
                            )
    m.setObjective(objective, GRB.MINIMIZE)

    # ограничения
    for v in range(n_vertices):
        m.addConstr(gp.quicksum(x[v][c] for c in range(n_colors)) == 1)

    for e, (i, j) in enumerate(edges):
        for c in range(n_colors):
            m.addConstr(x[i][c] + x[j][c] - y[e][c] <= 1)
    # print(m.getB)

    # m.write('acg-model.lp')
    m.setParam('TimeLimit', timeout)
    m.optimize()

    print(f"{m.ObjBound:g} <= objective <= {m.ObjVal:g}")
    return m.ObjBound, m.ObjVal
    # for v in range(n_vertices):
    #     for c in range(n_colors):
    #         val = m.getVarByName(f'x_{v}_{c}').X
    #         if val == 1:
    #             print(f'Vertex {v:3}: color {c}')


def gurobi_whith_clustering(main_graph: Graph, n_colors):
    n_main_edges = 0
    for i in main_graph.graph:
        n_main_edges += len(main_graph.graph[i])
    # n_main_edges=n_main_edges/2
    main_constrs = n_main_edges * n_colors + len(main_graph.graph)
    n_clusters = int(floor(main_constrs / 1600) + 2)
    # print('N_c ')
    # print(n_clusters)
    labels = clustering(main_graph, n_clusters)
    flag = False
    subgraphs = main_graph.split_graph_by_clusters(labels)
    while not flag:
        flag = True
        new_subs = []
        for i in subgraphs:
            n_constrs = i.get_n_edges() * n_colors * 2 + len(i.graph)
            if n_constrs > 1600:
                flag = False
                labels = clustering(i, 2, i.transform_graph_with_mapping()[0])
                new_subs.extend(i.split_graph_by_clusters(labels))
            else:
                new_subs.append(i)
        subgraphs = new_subs.copy()

    bound_val = [0, 0]
    for i in subgraphs:
        g_i = gurobi_optimisation(i, n_colors)
        bound_val[0] += g_i[0]
        bound_val[1] += g_i[1]

    return bound_val


if __name__ == '__main__':
    g = Graph()
    # g.init_by_n_v_mean_degree(1000,15)
    g.graph_generation_knn(160, 12)
    print(gurobi_whith_clustering(g, 4))
    print(g.get_obj(g.hill_climbing(4)[0]))
    print(g.get_obj(g.annealing(4, 10, 1.001)[0]))
