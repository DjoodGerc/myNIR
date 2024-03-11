import networkx as nx
from matplotlib import pyplot as plt
import random
from Graph_generation import Graph
from Greedy_alg import greedy_algorithm
from Annealing_alg import annealing_algorithm
from Greedy_alg import greedy_algorithm


"""
визуализация графа, получение ошибки метода
input: граф, кол-во цветов, метод (+для отжига начальная температура и скорость ее повышения)
??? поменять повышение на понижение ???
"""
def show_graph(G:Graph,n_colors,method,t=0.8,inc=1.0001):
    my_graph_class=G

    """ инициализация раскраски """
    if method==annealing_algorithm:
        painting= annealing_algorithm(G, n_colors, t, inc)
    else:
        painting=greedy_algorithm(G,n_colors)

    """
    библиотека для визуализации
    """
    g=nx.Graph()
    g.add_nodes_from(list(my_graph_class.graph.keys()))

    bind_colors={}
    """
    случайные цвета для визаулизации
    """
    for i in range(n_colors):

        r = lambda: random.randint(0, 255)
        col = '#%02X%02X%02X' % (r(), r(), r())

        while col in list(bind_colors.values()):

            col = '#%02X%02X%02X' % (r(), r(), r())

        bind_colors.update({i:col})
    """
    словарь сопоставления номера цвета и реального цвета для визаулизации
    """
    colors=[bind_colors[painting[i]] for i in list(my_graph_class.graph.keys())]

    """
        библиотека для визуализации
    """
    for i in range(len(list(my_graph_class.graph.keys()))):
        g.add_node(list(my_graph_class.graph.keys())[i],color=colors[i])


    """
    добавление ребер для визаулизации
    """
    my_graph=my_graph_class.graph
    edges_for_graph=[]
    for i in list(my_graph.keys()):
        arr=[]
        arr.extend((i,j) for j in list(my_graph[i].keys()))
        edges_for_graph.extend(arr)

    g.add_edges_from(edges_for_graph)


    nx.draw_circular(g,with_labels=True,node_color=colors)

    plt.show()
    return G.calculate_painting_error(G.painting)

g=Graph(7,5)
print(f"annealing_err: {show_graph(g,5,annealing_algorithm)}")
print(f"greedy_err: {show_graph(g,5,greedy_algorithm)}")
# g=Graph(10000,20)
# print(f"annealing_err: {show_graph(g,10,annealing_algorithm)}")
# print(f"greedy_err: {show_graph(g,10,greedy_algorithm)}")
