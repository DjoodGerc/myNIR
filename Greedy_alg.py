from Graph_generation import Graph
import numpy as np
"""
Засунуть в класс?

рефакторинг
"""

"""
нахождение цвета в который покрашено минимальное кол-во соседей
return 
"""
def find_min_color(color_and_n: dict) -> int:
    min_key = min(color_and_n, key=color_and_n.get)
    return min_key

"""
получение цвета для конкретной вершины
"""
def get_color_vertice(graph: dict, vertice: int, colors: list, painting: dict) -> int:
    bounded_vertices: dict = graph[vertice]
    # print(bounded_vertices)

    if len(list(bounded_vertices.keys()))==0:
        return np.random.randint(0,len(colors))
    else:

        """
        цвета в соседних вершинах
        """
        colors_in_sub = [painting[i] for i in list(bounded_vertices.keys())]

        """
        словарь {цвет:кол-во этого цвета среди соседей}
        """
        col_and_n={colors[i]:colors_in_sub.count(colors[i]) for i in colors}



        return find_min_color(col_and_n)


"""
Жадный алгоритм:
    проходимся по всем вершинам, красим вершину в цвет, 
    которого меньше всех среди соседей
"""
def greedy_algorithm(graph: Graph, n_colors: int)->dict:
    colors = [i for i in range(n_colors)]
    my_graph_dict: dict = graph.graph

    painting = {i: -1 for i in list(my_graph_dict)}
    graph.set_painting(painting)
    for i in list(graph.graph.keys()):
        vertice_color = get_color_vertice(graph.graph, i, colors, graph.painting)
        graph.painting.update({i: vertice_color})
    return painting



