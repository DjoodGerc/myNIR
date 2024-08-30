import random
# import matplotlib
import networkx as nx
from matplotlib import pyplot as plt

from Graph_generation import Graph
# matplotlib.use('Qt5Agg')
# from Graph_generation import errors_convergence

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

    nx.draw_circular(g, with_labels=True, node_color=colors)
    plt.suptitle(title)
    plt.show()

    return G.get_obj(G.coloring)


# print(f"annealing_err: {show_graph(g,5,annealing_algorithm)}")
# print(f"greedy_err: {show_graph(g,5,greedy_algorithm)}")
# g=Graph(10000,20)
# print(f"annealing_err: {show_graph(g,10,annealing_algorithm)}")
# print(f"greedy_err: {show_graph(g,10,greedy_algorithm)}")
if __name__ == '__main__':

    g = Graph(15, 7)
    rand_col=g.random_coloring(4)
    print(g.get_obj(rand_col))
    print(f"an: {show_graph(g, rand_col, 'random coloring')}")
    coloring_ann = g.annealing(4, 10, 1.034)[0]

    coloring_ann2 = g.annealing_snd(4, 10, 1.034)[0]

    coloring_hc = g.hill_climbing(4)[0]
    print()
    print(f"an: {show_graph(g, coloring_ann, 'annealing 1')}")
    print(f"an: {show_graph(g, coloring_ann2, 'annealing 2')}")
    print(f"hc: {show_graph(g, coloring_hc, 'hc')}")
    # g = Graph(300,100)
    # coloring_hc = g.hill_climbing(50)[0]
    # print(f"hc: {show_graph(g, coloring_hc, 'hc')}")
# TODO: графики сходимости obj/iterations np.minimum.accumulate;
#                          obj/temp or log(temp)
#                          10000+ VERT (20000 для 5g)