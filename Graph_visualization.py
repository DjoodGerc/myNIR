import random

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from g_coloring import g_one, g_two, g_three, g_four

from Graph_generation import Graph


# from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def show_graph(G, coloring: dict, title=""):
    """
    визуализация графа, получение ошибки метода
    input: граф, кол-во цветов, метод (+для отжига начальная температура и скорость ее повышения)
    ??? поменять повышение на понижение ???
    """

    my_graph_class = G
    n_colors = len(coloring)

    # библиотека для визуализации

    g = nx.Graph()
    g.add_nodes_from(list(my_graph_class.graph.keys()))

    bind_colors = {}

    # случайные цвета для визаулизации

    for i in range(n_colors):

        r = lambda: random.randint(0, 255)
        col = '#%02X%02X%02X' % (r(), r(), r())

        while col in list(bind_colors.values()):
            col = '#%02X%02X%02X' % (r(), r(), r())

        bind_colors.update({i: col})

    # словарь сопоставления номера цвета и реального цвета для визаулизации

    colors = [bind_colors[coloring[i]] for i in list(my_graph_class.graph.keys())]

    # библиотека для визуализации

    for i in range(len(list(my_graph_class.graph.keys()))):
        g.add_node(list(my_graph_class.graph.keys())[i], color=colors[i])

    # добавление ребер для визаулизации

    my_graph = my_graph_class.graph
    edges_for_graph = []
    for i in list(my_graph.keys()):
        arr = []
        arr.extend((i, j) for j in list(my_graph[i].keys()))
        edges_for_graph.extend(arr)

    g.add_edges_from(edges_for_graph)
    nx.draw(g,pos=G.pos, with_labels=True, node_color=colors)
    
    # nx.draw_circular(g, with_labels=True, node_color=colors)
    plt.suptitle(title)

    plt.show()

    return G.get_obj(coloring)



if __name__ == '__main__':
    n_v=350

    g = Graph(n_v, 3)
    
    g.coloring = None
    n_colors = 3

    rand_col = g.random_coloring(n_colors)
    g.graph_generation_knn(n_v,5)
    print(g.graph)
    print(g.pos)
    show_graph(g, rand_col)


    coloring_ann_snd = g.annealing_snd(n_colors, 10, 1.001, coloring=rand_col)[0]

    print(g.get_obj(coloring_ann_snd,True))

    show_graph(g,coloring_ann_snd)
    # print(f"{i} вершин; {i // 3} ср степень")
    # print(f"{n_colors} цветов")

    # print(f"случайная раскраска: {g.get_obj(rand_col)}")
    # for i in range(10,91,5):
    #     g = Graph(i, i//3)
    #     g.coloring=None
    #     n_colors=i//7+1
    #     rand_col=g.random_coloring(n_colors)
    #     print(f"{i} вершин; {i//3} ср степень")
    #     print(f"{n_colors} цветов")
    #
    #     print(f"случайная раскраска: {g.get_obj(rand_col)}")
    #
    #
    #     coloring_hc = g.get_obj(g.hill_climbing(n_colors,coloring=rand_col)[0])
    #     coloring_ann_fst = g.get_obj(g.annealing(n_colors,10,1.001,coloring=rand_col)[0])
    #     coloring_ann_snd = g.get_obj(g.annealing_snd(n_colors,10,1.001,coloring=rand_col)[0])
    #     g_one_col=g_one(g)
    #     g_two_col=g_two(g,n_colors)
    #     g_three_col=g_three(g,n_colors)
    #     g_four_col=g_four(g,n_colors)
    #     print(f"{n_colors} цветов")
    #     print(f"Hill Climbig: {str(coloring_hc)}")
    #     print(f"annealing 1: {coloring_ann_fst}")
    #     print(f"annealing 1: {coloring_ann_snd}")
    #     print("Новые Алгоритмы_____________")
    #     print(f"G1 (3 цвета): {g_one_col[1]}")
    #     print(f"G2: {str(g_two_col[1])}")
    #     print(f"G3: {str(g_three_col[1])}")
    #     print(f"G4: {str(g_four_col[1])}")
    #     print()
    #     print()
    #     print()
    #     # print(f"hc: {show_graph(g, g_three_col[0], 'hc')}")

# TODO: графики сходимости obj/iterations np.minimum.accumulate;
#                          obj/temp or log(temp)
#                          10000+ VERT (20000 для 5g)