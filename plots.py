import networkx as nx
from matplotlib import pyplot as plt
import random
from Graph_generation import Graph
from Greedy_alg import greedy_algorithm
from Annealing_alg import annealing_algorithm
from Greedy_alg import greedy_algorithm



def show_graph(G,coloring: dict,title=""):
    """
    визуализация графа, получение ошибки метода
    input: граф, кол-во цветов, метод (+для отжига начальная температура и скорость ее повышения)
    ??? поменять повышение на понижение ???
    """

    my_graph_class=G
    n_colors= len(coloring)



    # библиотека для визуализации

    g=nx.Graph()
    g.add_nodes_from(list(my_graph_class.graph.keys()))

    bind_colors={}

    # случайные цвета для визаулизации

    for i in range(n_colors):

        r = lambda: random.randint(0, 255)
        col = '#%02X%02X%02X' % (r(), r(), r())

        while col in list(bind_colors.values()):

            col = '#%02X%02X%02X' % (r(), r(), r())

        bind_colors.update({i:col})

    # словарь сопоставления номера цвета и реального цвета для визаулизации

    colors=[bind_colors[coloring[i]] for i in list(my_graph_class.graph.keys())]


    # библиотека для визуализации

    for i in range(len(list(my_graph_class.graph.keys()))):
        g.add_node(list(my_graph_class.graph.keys())[i],color=colors[i])



    # добавление ребер для визаулизации

    my_graph=my_graph_class.graph
    edges_for_graph=[]
    for i in list(my_graph.keys()):
        arr=[]
        arr.extend((i,j) for j in list(my_graph[i].keys()))
        edges_for_graph.extend(arr)

    g.add_edges_from(edges_for_graph)


    nx.draw_circular(g,with_labels=True,node_color=colors)
    plt.suptitle(title)
    plt.show()

    return G.get_obj(G.coloring)


# print(f"annealing_err: {show_graph(g,5,annealing_algorithm)}")
# print(f"greedy_err: {show_graph(g,5,greedy_algorithm)}")
# g=Graph(10000,20)
# print(f"annealing_err: {show_graph(g,10,annealing_algorithm)}")
# print(f"greedy_err: {show_graph(g,10,greedy_algorithm)}")
if __name__ == '__main__':
    g = Graph(10,8)


    coloring_ann=g.annealing(5, 0.85, 1.0001)
    print(f"an: {show_graph(g, coloring_ann,'annealing')}")
    coloring_hc=g.hill_climbing(5)

    print(f"hc: {show_graph(g, coloring_hc, 'hc')}")



