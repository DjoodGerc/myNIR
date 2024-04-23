import numpy as np
# from plots import show_graph

class Graph:
    coloring = {}
    graph = {}

    def __init__(self, n_vertices: int, mean_degree: int):
        self.graph_generation(n_vertices, mean_degree)

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

    def get_obj(self, coloring) -> float:
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
                    already_checked[i].append(j)

        return err

    def graph_generation(self, n_vertice: int, mean_degree: int) -> dict:
        """
            генерация графа
        """
        rs = np.random.RandomState(125156)
        graph = {i: {} for i in range(n_vertice)}

        n_edges = mean_degree * n_vertice / 2
        while n_edges > 0:

            while True:
                v1 = rs.randint(n_vertice)
                v2 = (rs.randint(1, n_vertice) + v1) % n_vertice
                if not v2 in graph[v1]:
                    break

            weight = rs.random()

            graph[v1][v2] = weight
            graph[v2][v1] = weight

            n_edges -= 1
        # print(f"error: {sum(map(len, graph.values()))}")
        self.graph = graph

        return graph

    def random_coloring(self, n_colors: int, seed: int):
        rs = np.random.RandomState(seed)
        return {i: rs.randint(n_colors) for i in self.graph}

    def create_err_table(self, n_colors):
        #get obj
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
            err_table[i][old_color] -=self.graph[vertice][i]
            err_table[i][new_color] += self.graph[vertice][i]
        return err_table

    def hill_climbing(self, n_colors: int):

        g = self.graph
        self.coloring = self.random_coloring(n_colors, 2567)
        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        k = 0
        while True:

            new_colors = np.argmin(err_table, axis=1)
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]
            v_candidate = np.argmin(delta_array)
            d = delta_array[v_candidate]
            selected_color = new_colors[v_candidate]

            if d >= 0 or k > 50000:

                break


            err_table = self.change_err_table(err_table, v_candidate, selected_color)
            self.coloring[v_candidate] = selected_color

            k += 1
        return self.coloring

    def annealing(self, n_colors: int, t: float, inc: float):
        g = self.graph
        self.coloring = self.random_coloring(n_colors, 12453)
        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        # print(g)
        # print(err_table)

        while 1 / t > 10 ** -6:

            # new_colors = np.argmin(err_table, axis=1)
            new_colors=[]
            len_arr=len(err_table[0])
            for i in range(len(err_table)):
                new_colors.append(np.random.randint(len_arr))

            # кол-во соседей нового цвета - кол-во соседей старого цвета
            # print(new_colors)
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]

            # ?
            # v_candidate = np.random.randint(len(delta_array))

            v_candidate = np.argmin(delta_array)
            d = delta_array[v_candidate]
            r = np.random.rand()
            selected_color = new_colors[v_candidate]

            # print(d)
            if d <= 0 or r < np.exp(-t * d):
                err_table = self.change_err_table(err_table, v_candidate, selected_color)
                self.coloring[v_candidate] = selected_color


            t *= inc

        return self.coloring


if __name__ == '__main__':
    a = Graph(5, 3)

    # a.create_err_table(5)

    print(a.hill_climbing(3))
    # a.random_coloring(5)

    # print(check(graph_generation(3, 33)))
