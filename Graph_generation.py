import numpy as np


# from plots import show_graph

class Graph:
    rs = np.random.RandomState(42)
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
        # print(f"error: {sum(map(len, graph.values()))}")
        self.graph = graph

        return graph

    def random_coloring(self, n_colors: int):

        return {i: self.rs.randint(n_colors) for i in self.graph}

    def create_err_table(self, n_colors):
        # get obj
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
    def find_min_not_eq_this(self,delta_arr,new_colors):
        this_min_not_eq=delta_arr[0]
        tmine_arg=None
        if new_colors[0]==self.coloring[0]:
            q_arg=0
        else:
            q_arg=None
        for i in range(1,len(delta_arr)):
            if new_colors[i]==self.coloring[i]:
                q_arg=i
            else:
                if delta_arr[i]<this_min_not_eq:
                    this_min_not_eq=delta_arr[i]
                    tmine_arg=i
        return tmine_arg, this_min_not_eq,q_arg

    def hill_climbing(self, n_colors: int,seed=77):

        g = self.graph
        self.coloring = self.random_coloring(n_colors)
        fst_o=self.get_obj(self.coloring)
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

            if d >= 0 :#or k > 50000:
                break
            fst_o+=d
            err_table = self.change_err_table(err_table, v_candidate, selected_color)
            self.coloring[v_candidate] = selected_color

            k += 1
            obj_array.append(obj_array[len(obj_array)-1]+d)
            print(k)
        # print(fst_o)
        return self.coloring,obj_array

    def annealing(self, n_colors: int, t: float, inc: float):
        g = self.graph
        self.coloring = self.random_coloring(n_colors)
        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        # print(g)
        # print(err_table)
        obj_array=[]
        obj_array.append(self.get_obj(self.coloring))
        temp_array=[]
        # print(err_table)

        while t > 10 ** -5:

            # new_colors = np.argmin(err_table, axis=1)
            new_colors = []
            len_arr = len(err_table[0])
            for i in range(len(err_table)):
                # new_colors.append((np.random.randint(1,len_arr)+self.coloring[i])%len_arr)
                new_colors.append(self.rs.randint(len_arr))

            # кол-во соседей нового цвета - кол-во соседей старого цвета
            # print(new_colors)
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]

            # ?
            # v_candidate = np.random.randint(len(delta_array))

            # v_candidate = np.argmin(delta_array)
            fmne = self.find_min_not_eq_this(delta_array,new_colors)
            # print(fmne)
            if fmne[0] is None:
                v_candidate = fmne[2]
            elif fmne[1]<0:
                # print(v_candidate==fmne[0])
                v_candidate=fmne[0]
            elif fmne[2] is None :
                v_candidate = fmne[0]
            else:
                r=self.rs.random()
                if r < np.exp(-fmne[1]/t):

                    v_candidate=fmne[0]
                else:
                    v_candidate=fmne[2]

            if v_candidate is None:
                v_candidate=0
            d = delta_array[v_candidate]

            r = self.rs.random()
            # print(fmne)

            selected_color = new_colors[v_candidate]

            if d < 0 or r < np.exp(-d/t):
                err_table = self.change_err_table(err_table, v_candidate, selected_color)
                self.coloring[v_candidate] = selected_color
                obj_array.append(obj_array[len(obj_array)-1]+d)
            else:
                obj_array.append(obj_array[len(obj_array)-1])

            # print(new_colors)
            # if(d>=0):
            #     print(f"d: {d}")
            #     print(f"exp: {np.exp(-d/t)}")
            t /= inc
            print(f"fst: {t}")


        return self.coloring,obj_array



    def annealing_snd(self, n_colors: int, t: float, inc: float,seed=77):
        g = self.graph
        self.coloring = self.random_coloring(n_colors)
        err_table = self.create_err_table(n_colors)
        n_vertices = len(g)
        # print(g)
        # print(err_table)
        obj_array=[]
        obj_array.append(self.get_obj(self.coloring))
        temp_array=[]
        # print(err_table)

        while t > 10 ** -5:

            # new_colors = np.argmin(err_table, axis=1)
            new_colors = []
            len_arr = len(err_table[0])
            for i in range(len(err_table)):
                new_colors.append((self.rs.randint(1,len_arr)+self.coloring[i])%len_arr)
                # new_colors.append(np.random.randint(len_arr))

            # кол-во соседей нового цвета - кол-во соседей старого цвета
            # print(new_colors)
            delta_array = err_table[np.arange(n_vertices), new_colors] - err_table[
                np.arange(n_vertices), list(self.coloring.values())]

            # ?
            # v_candidate = np.random.randint(len(delta_array))

            v_candidate = np.argmin(delta_array)


            d = delta_array[v_candidate]
            # print(d)
            # print(t)
            # print(delta_array)
            r = self.rs.random()
            # print(v_candidate)
            selected_color = new_colors[v_candidate]
            # print(selected_color==self.coloring[v_candidate])
            # print(d)
            if d < 0 or r < np.exp(-d/t):
                err_table = self.change_err_table(err_table, v_candidate, selected_color)
                self.coloring[v_candidate] = selected_color
                obj_array.append(obj_array[len(obj_array) - 1] + d)
            else:
                obj_array.append(obj_array[len(obj_array) - 1])
            # print(new_colors)
            # if(d>=0):
            #     print(f"d: {d}")
            #     print(f"exp: {np.exp(-d/t)}")
            t /= inc
            print(f"snd: {t}")
        return self.coloring,obj_array



if __name__ == '__main__':
    a = Graph(5, 3)

    # a.create_err_table(5)

    # print(a.hill_climbing(3))
    # a.random_coloring(5)

    # print(check(graph_generation(3, 33)))
