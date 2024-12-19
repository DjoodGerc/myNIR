import numpy as np
from networkx.classes import non_neighbors

from sklearn.neighbors import NearestNeighbors


# from plots import show_graph

class Graph:
    rs = np.random.RandomState(42)
    coloring = {}
    graph = {}
    pos = []

    def get_graph(self):
        return self.graph

    def __init__(self, n_vertices: int, mean_degree: int, coloring=None, graph=None):
        if coloring is None or Graph is None:
            self.graph_generation(n_vertices, mean_degree)
        else:
            self.coloring = coloring
            self.graph = graph

    def get_d_obj(self, vertice, color):
        """
             ошибка при изменении одного цвета
             изменение цвета вершины влияет только на соседей
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

    def graph_generation_knn(self, n_vertice: int, mean_degree: int):
        dots = [[self.rs.rand() * 10, self.rs.rand() * 10] for i in range(n_vertice)]
        self.pos = dots
        nbrs = NearestNeighbors(n_neighbors=mean_degree + 1, algorithm='ball_tree').fit(dots)
        distances, indices = nbrs.kneighbors(dots)
        graph = {i: {} for i in range(n_vertice)}
        print(distances)
        print(indices)
        n_edges = mean_degree * n_vertice / 2

        for i in range(len(indices)):
            n_neighbors = mean_degree - len(graph[i])
            i_neigh = indices[i].tolist()
            i_dist = distances[i].tolist()
            j = 0
            i_neigh.remove(i)
            i_dist.remove(0.)
            while j < n_neighbors and len(i_neigh) > 0:

                v2 = self.rs.choice(i_neigh)
                ind = i_neigh.index(v2)
                i_neigh.remove(v2)

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
            генерация графа
        """

        graph = {i: {} for i in range(n_vertice)}

        n_edges = mean_degree * n_vertice / 2
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

            if d >= 0:  # or k > 50000:
                break
            fst_o += d
            err_table = self.change_err_table(err_table, v_candidate, selected_color)
            self.coloring[v_candidate] = selected_color

            k += 1
            obj_array.append(obj_array[len(obj_array) - 1] + d)
        return self.coloring, obj_array

    def annealing(self, n_colors: int, t: float, inc: float, coloring=None):
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


if __name__ == '__main__':
    a = Graph(5, 3)
