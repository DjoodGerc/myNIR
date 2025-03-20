import numpy as np
from networkx.classes import non_neighbors

from sklearn.neighbors import NearestNeighbors



# from plots import show_graph

class Graph:
    rs = np.random.RandomState(42)
    n_vertices = 0
    mean_degree = 0
    coloring = {}
    graph = {}
    pos = {}

    def get_graph(self):

        return self.graph


    def init_by_graph(self, graph: dict, pos: dict):
        self.graph = dict(sorted(graph.items()))
        self.n_vertices = len(graph)
        # print(pos)
        self.pos = dict(sorted(pos.items()))

    def init_by_n_v_mean_degree(self, n_vertices: int, mean_degree: int, coloring=None, graph=None):
        """

        :param n_vertices: кол-во вершин
        :param mean_degree: средняя степень вершины
        :param coloring: раскраска
        :param graph: граф

        если coloring или graph == None - генерируем новый граф
        иначе присваиваем раскраску и граф.
        """
        self.n_vertices = n_vertices
        self.mean_degree = mean_degree
        if coloring is None or Graph is None:
            self.graph_generation(n_vertices, mean_degree)
        else:
            self.coloring = coloring
            self.graph = graph

    def get_d_obj(self, vertice, color):
        """
        ошибка при изменении одного цвета
        изменение цвета вершины влияет только на соседей


        :param vertice: вершина
        :param color: цвет
        :return: delta, новая ошибка, старая ошибка
        """

        keys = self.graph[vertice].keys()
        err_old = 0
        for i in keys:
            if self.coloring[vertice] == self.coloring[i]:
                err_old += self.graph[vertice][i]
        err_new: int = 0
        for i in keys:
            if color == self.coloring[i]:
                err_new += self.graph[vertice][i]

        delta = (err_new - err_old)
        return delta, err_new, err_old

    def get_obj(self, coloring, print_flag=False) -> float:
        """
       вычисление полной ошибки раскраски
       !оптимизировать!
        """
        err = 0
        already_checked = {i: [] for i in self.graph.keys()}

        for i in self.graph.keys():
            this_ver_color = coloring[i]
            for j in self.graph[i].keys():
                if not i in already_checked[j]:
                    if this_ver_color == coloring[j]:
                        err += self.graph[i][j]
                        if print_flag == True:
                            print(i, j, self.graph[i][j])
                    already_checked[i].append(j)

        return err
    def get_n_edges(self):
        n_edges=0
        for i in self.graph:
            n_edges+=len(self.graph[i])
        return n_edges/2

    def graph_generation_knn(self, n_vertice: int, mean_degree: int) -> dict:
        """
        генерация графов, используя k ближайших соседей + pos
        :param n_vertice:
        :param mean_degree:
        :return:
        """
        # назначаем каждой точки случайные координаты
        dots = {i: [self.rs.rand() * 10, self.rs.rand() * 10] for i in range(n_vertice)}
        self.pos = dots
        # ищем mean degree + 1 ближайших соседей для каждой точки
        dots_arr=list(dots.values())
        nbrs = NearestNeighbors(n_neighbors=mean_degree + 1, algorithm='ball_tree').fit(dots_arr)
        # ищем дистанцию и индексы соседей
        distances, indices = nbrs.kneighbors(dots_arr)
        graph = {i: {} for i in range(n_vertice)}
        # print(distances)
        # print(indices)
        n_edges = mean_degree * n_vertice / 2
        # пробегаемся по массиву соседей для каждой вершины, назначаем связи
        for i in range(len(indices)):

            n_neighbors = mean_degree - len(graph[i])
            i_neigh = indices[i].tolist()
            i_dist = distances[i].tolist()
            j = 0
            i_neigh.remove(i)
            i_dist.remove(0.)
            while j < n_neighbors and len(i_neigh) > 0:
                # выбираем случайно вершину, если связи еще нет - создаем ее

                v2 = self.rs.choice(i_neigh)
                ind = i_neigh.index(v2)
                i_neigh.remove(v2)
                # TODO: спросить нужно ли учитывать дистанцию
                # if i_dist[ind] > 0.5:
                #     continue

                # i_neigh.remove(i)
                if v2 in graph[i]:
                    j += 1
                    continue
                else:
                    weight = self.rs.random()

                    graph[i][v2] = weight
                    graph[v2][i] = weight
                    j += 1

        self.graph = graph

        return graph

    def graph_generation(self, n_vertice: int, mean_degree: int) -> dict:
        """
            генерация графа (случайные соседи)
        """

        graph = {i: {} for i in range(n_vertice)}

        n_edges = mean_degree * n_vertice / 2
        # каждой вершине назначем mean_degree случайных соседей
        while n_edges > 0:

            while True:
                v1 = self.rs.randint(n_vertice)
                v2 = (self.rs.randint(1, n_vertice) + v1) % n_vertice
                if not v2 in graph[v1]:
                    break

            weight = self.rs.random()

            graph[v1][v2] = weight
            graph[v2][v1] = weight

            n_edges -= 1
        self.graph = graph

        return graph

    def random_coloring(self, n_colors: int):

        return {i: self.rs.randint(n_colors) for i in self.graph}

    def create_err_table(self, n_colors):
        """
        таблица ошибок
        :param n_colors:
        :return:
        представляет собой таблицу, где вертикально - вершины, горизонтально -цвета
        в каждой клетке сумма соседей vi вершины ck цвета
        """
        g = self.graph
        err_table = []

        for i in g:
            line = np.zeros(n_colors).tolist()
            for j in g[i]:
                v_color = self.coloring[j]

                line[v_color] += g[i][j]
            err_table.append(line)

        err_table = np.array(err_table)
        return err_table

    def change_err_table(self, err_table, vertice, new_color):
        # замена таблицы цветов, после смены цвета вершины
        neighbors = list(self.graph[vertice].keys())
        old_color = self.coloring[vertice]

        for i in neighbors:
            err_table[i][old_color] -= self.graph[vertice][i]
            err_table[i][new_color] += self.graph[vertice][i]
        return err_table

    def find_min_not_eq_this(self, delta_arr, new_colors):
        this_min_not_eq = delta_arr[0]
        tmine_arg = None
        if new_colors[0] == self.coloring[0]:
            q_arg = 0
        else:
            q_arg = None
        for i in range(1, len(delta_arr)):
            if new_colors[i] == self.coloring[i]:
                q_arg = i
            else:
                if delta_arr[i] < this_min_not_eq:
                    this_min_not_eq = delta_arr[i]
                    tmine_arg = i
        return tmine_arg, this_min_not_eq, q_arg

    def hill_climbing(self, n_colors: int, seed=77, coloring=None):
        """
        восхождение на гору
        :param n_colors:
        :param seed:
        :param coloring:
        :return:
        определяем случайную раскраску, ищем вершину с наибольшим числом конфликтов, меняем ей цвет, на тот, который даст большую разницу.
        продолжаем, пока замена цвета не перестанет давать результат
        """
        g = self.graph
        if coloring is None:
            self.coloring = self.random_coloring(n_colors)
        elif coloring is not None:
            self.coloring = coloring
        fst_o = self.get_obj(self.coloring)
        err_table = self.create_err_table(n_colors)

        n_vertices = len(g)
        k = 0

        obj_array = []
        obj_array.append(self.get_obj(self.coloring))

        while True:

            new_colors = np.argmin(err_table, axis=1)
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]
            v_candidate = np.argmin(delta_array)

            d = delta_array[v_candidate]
            selected_color = new_colors[v_candidate]
            # условие остановке, по невозможности поменять цвет либо по кол-ву операций
            if d >= 0:  # or k > 50000:
                break
            fst_o += d
            err_table = self.change_err_table(err_table, v_candidate, selected_color)
            self.coloring[v_candidate] = selected_color

            k += 1
            obj_array.append(obj_array[len(obj_array) - 1] + d)
        return self.coloring, obj_array

    def annealing(self, n_colors: int, t: float, inc: float, coloring=None):
        """
        отжиг
        :param n_colors:
        :param t:
        :param inc:
        :param coloring:
        :return:
        ищем лучшую вершину для замены цвета при помощи таблицы ошибок, меняем ей цвет с некоторой вероятностью
        продолжаем, пока температура не упадет достаточно низко
        """
        g = self.graph
        if coloring is None:
            self.coloring = self.random_coloring(n_colors)
        elif coloring is not None:
            self.coloring = coloring
        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        obj_array = []
        obj_array.append(self.get_obj(self.coloring))

        while t > 10 ** -5:

            new_colors = []
            len_arr = len(err_table[0])

            for i in range(len(err_table)):
                # new_colors.append((np.random.randint(1,len_arr)+self.coloring[i])%len_arr)
                new_colors.append(self.rs.randint(len_arr))

            # кол-во соседей нового цвета - кол-во соседей старого цвета
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]

            # ?
            # v_candidate = np.random.randint(len(delta_array))

            fmne = self.find_min_not_eq_this(delta_array, new_colors)
            if fmne[0] is None:
                v_candidate = fmne[2]
            elif fmne[1] < 0:
                v_candidate = fmne[0]
            elif fmne[2] is None:
                v_candidate = fmne[0]
            else:
                r = self.rs.random()
                if r < np.exp(-fmne[1] / t):

                    v_candidate = fmne[0]
                else:
                    v_candidate = fmne[2]

            if v_candidate is None:
                v_candidate = 0
            d = delta_array[v_candidate]

            r = self.rs.random()

            selected_color = new_colors[v_candidate]

            if d < 0 or r < np.exp(-d / t):

                err_table = self.change_err_table(err_table, v_candidate, selected_color)
                self.coloring[v_candidate] = selected_color
                obj_array.append(obj_array[len(obj_array) - 1] + d)
            else:
                obj_array.append(obj_array[len(obj_array) - 1])

            t /= inc

        return self.coloring, obj_array

    def annealing_snd(self, n_colors: int, t: float, inc: float, seed=77, coloring=None):
        """
        отжиг 2 моя версия наподумать на будущее (в диплом не идет)
        :param n_colors:
        :param t:
        :param inc:
        :param seed:
        :param coloring:
        :return:
        """

        g = self.graph
        # print(coloring)
        # print(coloring is None)
        if coloring is None:
            self.coloring = self.random_coloring(n_colors)
        elif coloring is not None:
            self.coloring = coloring

        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        # print(g)
        # print(err_table)
        obj_array = []
        obj_array.append(self.get_obj(self.coloring))
        temp_array = []

        while t > 10 ** -5:

            new_colors = []
            len_arr = len(err_table[0])

            for i in range(len(err_table)):
                new_colors.append((self.rs.randint(1, len_arr) + self.coloring[i]) % len_arr)

            # кол-во соседей нового цвета - кол-во соседей старого цвета
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]

            # ?
            # v_candidate = np.random.randint(len(delta_array))

            v_candidate = np.argmin(delta_array)

            d = delta_array[v_candidate]
            r = self.rs.random()
            selected_color = new_colors[v_candidate]
            if d < 0 or r < np.exp(-d / t):

                err_table = self.change_err_table(err_table, v_candidate, selected_color)
                self.coloring[v_candidate] = selected_color
                obj_array.append(obj_array[len(obj_array) - 1] + d)
            else:
                obj_array.append(obj_array[len(obj_array) - 1])

            t /= inc

        return self.coloring, obj_array

    def get_edges_arr(self):
        arr = []
        for i in self.graph:
            for j in self.graph[i]:
                element = []
                if ([self.graph[i][j], [j, i]]) not in arr:
                    arr.append([self.graph[i][j], [i, j]])
        return arr

    def to_adjacency_matrix(self,edges_tr=None) -> np.array:
        matrix = np.zeros((len(self.graph), len(self.graph)))
        if edges_tr is None:

            for i in self.graph:
                for j in self.graph[i].keys():
                    matrix[i][j] = self.graph[i][j]
        else:
            for i, j in edges_tr:
                matrix[i][j]=edges_tr[i,j]
        return matrix
    def decomposition(self, labels):
        unique = np.unique(labels)
        clusters = [[] for i in unique]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        print(self.graph)
        print(clusters)

    def split_graph_by_clusters(self, labels):
        """
        Splits a graph into subgraphs based on the given clusters.

        Args:
          graph: A dictionary representing the graph. Keys are node indices,
                 and values are dictionaries mapping neighbor indices to edge weights.
          clusters: A list of lists, where each inner list represents a cluster
                    containing node indices.

        Returns:
          A list of subgraphs, where each subgraph is a dictionary representing
          the connections within that cluster.
        """
        unique = np.unique(labels)
        clusters = [[] for i in unique]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        print(len(clusters))
        subgraphs = []
        print(clusters)
        for cluster in clusters:
            subgraph = {}
            p = {}
            for node in cluster:

                # p[node]=self.pos[node].copy()
                subgraph[node] = {}  # Initialize node in subgraph
                for neighbor, weight in self.graph.get(node, {}).items():
                    if neighbor in cluster:  # Only include connections within the cluster
                        subgraph[node][neighbor] = weight

            new_sub=Graph()
            new_sub.init_by_graph(subgraph, p)
            subgraphs.append(new_sub)

        return subgraphs

    def transform_graph_with_mapping(self):
        """
        Transforms a graph represented as a dictionary of dictionaries into a
        dictionary where keys are tuples of (node1, node2) and values are edge weights.
        Additionally, remaps node indices to a range from 0 to len(graph_data) - 1
        and returns the mapping between new and old node indices.

        Args:
          graph_data: A dictionary representing the graph. Keys are node indices,
                      and values are dictionaries mapping neighbor indices to edge weights.

        Returns:
          A tuple containing:
            - A dictionary representing the transformed graph with remapped node indices.
            - A dictionary representing the mapping from new node indices to old node indices.
        """

        transformed_graph = {}
        old_to_new_mapping = {node: i for i, node in
                              enumerate(sorted(self.graph.keys()))}  # Create a mapping to new indices
        new_to_old_mapping = {i: node for node, i in old_to_new_mapping.items()}  # Invert the mapping

        for original_node1, neighbors in self.graph.items():
            node1 = old_to_new_mapping[original_node1]  # Get the new index for node1

            for original_node2, weight in neighbors.items():
                node2 = old_to_new_mapping[original_node2]  # Get the new index for node2
                # Ensure consistent ordering in the tuple key (node1 < node2)

                key_1 = (node1, node2)

                key_2 = (node2, node1)
                transformed_graph[key_1] = weight / 2
                transformed_graph[key_2] = weight / 2

        return transformed_graph, old_to_new_mapping


if __name__ == '__main__':
    a = Graph(30, 3)
