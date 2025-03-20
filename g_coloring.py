from collections import OrderedDict

import numpy as np

from Graph_generation import Graph

"""очень грустный алгоритм. шансов нет. только для 3 цветов """


def g_one(graph: Graph):
    """
    Сортируем вершины по убываению суммы весов инцидентных ребер. берем первый
    красим в первый цвет, всех соседей красим в 2 и 3 цвет, продолжаем по непокрашенным.

    :param graph: исходный граф
    :return: [раскраска, ошибка]
    """
    sorted_vertices = OrderedDict()
    for i in graph.graph:

        wSum = 0

        for j in graph.graph[i]:
            wSum += graph.graph[i][j]
        sorted_vertices[i] = wSum
    sorted_vertices = OrderedDict(sorted(sorted_vertices.items(), key=lambda item: item[1], reverse=True))

    while len(sorted_vertices) != 0:
        v = sorted_vertices.popitem(last=False)[0]
        graph.coloring[v] = 0
        t = 0
        for i in graph.graph[v]:
            try:
                sorted_vertices.pop(i)
            except:
                continue
            if t % 2 == 0:
                graph.coloring[i] = 1
            else:
                graph.coloring[i] = 2
            t += 1
    return [graph.coloring, graph.get_obj(graph.coloring)]


def g_two(graph: Graph, n_colors: int):
    """

    :param graph: исходный граф
    :param n_colors: кол-во цветов
    :return: [раскраска,ошибка]
    """

    def change_err_table(vertice: int, color: int):
        """
        меняем таблицу ошибок, убираем уже покрашенную вершину,
        перезаполняем соседей (непокрашенных)

        :param vertice: id вешины
        :param color: цвет вершины
        :return: void
        """
        err_table.pop(not_colored_vs.index(vertice))
        not_colored_vs.remove(vertice)
        for i in graph.graph[vertice]:
            if i in not_colored_vs:
                err_table[not_colored_vs.index(i)][color] += graph.graph[vertice][i]

    def create_delta_arr():
        """
        получаем вершину и цвет для перекрашивания, анализируя таблицу ошибок
        :return: [вершина, цвет]
        """
        # выбираем каждоый вершине новый цвет (тот у сумма весов ребер инцидентных определенному цвету минимальная)
        new_colors = np.argmin(err_table, axis=1)
        weights = [err_table[i][new_colors[i]] for i in range(len(new_colors))]
        id = np.argmin(weights)
        nc = new_colors[id]
        nv = not_colored_vs[id]
        # for i in range(len(err_table)):
        #     if graph.coloring[not_colored_vs[i]]==-1:
        #         delta_array.append(0)
        #         continue
        #     else:
        #
        #         new=err_table[i,new_colors[i]]
        #         old=err_table[i,graph.coloring(not_colored_vs[i])]
        #         delta_array.append(new-old)
        # v_candidate = not_colored_vs[np.argmin(delta_array)]
        # c_candidate = new_colors[not_colored_vs.index(v_candidate)]
        # graph.coloring[v_candidate]=c_candidate.item()
        # print(delta_array)
        # return [v_candidate,c_candidate]
        graph.coloring[nv] = nc.item()
        return [nv, nc]

    """
    создаем таблицу ошибок. если вершина не покрашена сопоставляем ей -1 цвет.
    каждой вершине сопостовляем сумму весов ребер соседей одинакового цвета. 
    """
    not_colored_vs = [i for i in graph.graph]
    line = [0 for i in range(n_colors + 1)]
    err_table = [line.copy() for i in range(len(graph.graph))]
    graph.coloring = {i: -1 for i in graph.graph}
    while len(err_table) != 0:
        # красим вершины пока они не кончатся
        res = create_delta_arr()
        change_err_table(res[0], res[1])

    return [graph.coloring, graph.get_obj(graph.coloring)]


def g_three(graph: Graph, n_colors: int):
    """
    G3: сортируем ребра в порядке невозрастания весов и красим каждый
    :param graph:
    :param n_colors:
    :return:
    """
    line = [0 for i in range(n_colors + 1)]
    err_table = [line.copy() for i in range(len(graph.graph))]

    def color_v(vertice: int, color: int):
        for i in graph.graph[vertice]:
            err_table[i][color] += graph.graph[vertice][i]

    # получаем отсортированные ребра
    sorted_ea = edges_arr = graph.get_edges_arr()

    sorted_ea.sort(key=lambda arr: arr[0], reverse=True)

    graph.coloring = {i: -1 for i in graph.graph}
    # красим
    for i in sorted_ea:
        for j in i[1]:
            if graph.coloring[j] == -1:
                # TODO: здесь без if работает лучше - проверь
                color = np.argmin(err_table[j]).item()
                color_v(j, color)
                graph.coloring[j] = color
    return [graph.coloring, graph.get_obj(graph.coloring)]


def g_four(graph: Graph, n_colors: int):



    line = [0 for i in range(n_colors + 1)]
    err_table = [line.copy() for i in range(len(graph.graph))]

    def color_v(vertice: int, color: int):
        for i in graph.graph[vertice]:
            err_table[i][color] += graph.graph[vertice][i]

    vertices_arr=[]
    for i in graph.graph:
        sum_e=0
        for j in graph.graph[i]:
            sum_e+=graph.graph[i][j]
        vertices_arr.append([sum_e,i])
    vertices_arr.sort(key=lambda arr: arr[0],reverse=True)
    graph.coloring = {i: -1 for i in graph.graph}
    # красим
    for i in vertices_arr:
            color = np.argmin(err_table[i[1]]).item()
            color_v(i[1], color)
            graph.coloring[i[1]] = color
    return [graph.coloring,graph.get_obj(graph.coloring)]




if __name__ == '__main__':
    g = Graph(16, 6)
    # print(g_one(g))
    # print(g_two(g, 3))
    # print(g_three(g, 3))
    print(g_four(g,3))
    # print(g.get_edges_arr())
    # h=g.get_edges_arr()
    # h.sort(key=lambda arr: arr[0],reverse=True)
    # print(h)
