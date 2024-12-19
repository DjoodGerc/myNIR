import numpy as np

from Graph_generation import Graph

"""
Засунуть в класс?

рефакторинг
"""


def find_min_color(color_and_n: dict) -> int:
    """
    нахождение цвета в который покрашено минимальное кол-во соседей
    return
    """
    min_key = min(color_and_n, key=color_and_n.get)
    return min_key


def get_color_vertice(graph: dict, vertice: int, colors: list, coloring: dict) -> int:
    """
    получение цвета для конкретной вершины
    """
    bounded_vertices: dict = graph[vertice]
    # print(bounded_vertices)

    if len(list(bounded_vertices.keys())) == 0:
        return np.random.randint(0, len(colors))
    else:

        # цвета в соседних вершинах

        colors_in_sub = [coloring[i] for i in list(bounded_vertices.keys())]

        # словарь {цвет:кол-во этого цвета среди соседей}

        col_and_n = {colors[i]: colors_in_sub.count(colors[i]) for i in colors}

        return find_min_color(col_and_n)


def greedy_algorithm(graph: Graph, n_colors: int) -> dict:
    """
    Жадный алгоритм:
        проходимся по всем вершинам, красим вершину в цвет, которого меньше всех среди соседей
    """

    colors = [i for i in range(n_colors)]
    my_graph_dict: dict = graph.graph

    coloring = {i: -1 for i in list(my_graph_dict)}
    graph.coloring = coloring
    for i in list(my_graph_dict.keys()):
        vertice_color = get_color_vertice(my_graph_dict, i, colors, graph.coloring)
        graph.coloring[i] = vertice_color
    return coloring
