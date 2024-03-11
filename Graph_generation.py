import numpy as np



class Graph:


    painting = {}
    graph = {}

    def set_painting(self,painting: painting):
        self.painting=painting

    def __init__(self,n_vertice:int,mean_degree:int):
        self.graph_generation(n_vertice,mean_degree)


    """
     ошибка при изменении одного цвета
     изменение цвета вершины влияет только на соседей
    """
    def find_differents_painting_with_one_changed_color(self,vertice,color):
        keys=self.graph[vertice].keys()
        err_old=0
        for i in keys:
            if self.painting[vertice]==self.painting[i]:
                err_old+=1
        err_new:int =0
        for i in keys:
            if color == self.painting[i]:
                err_new += 1

        delta=(err_new-err_old)
        return delta,err_new,err_old


    """
    вычисление полной ошибки раскраски

    !оптимизировать!
    """
    def calculate_painting_error(self,painting=painting)->float:
        err=0
        already_checked={i:[] for i in self.graph.keys()}

        for i in self.graph.keys():
            this_ver_color=painting[i]
            for j in self.graph[i].keys():
                if not i in already_checked[j]:
                    if this_ver_color==painting[j]:
                        err+=1
                    already_checked[i].append(j)


        return err


    """
    генерация графа
    """
    def graph_generation(self,n_vertice: int , mean_degree: int)->dict:
        rs = np.random.RandomState(42)
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
        print(f"error: {sum(map(len,graph.values()))}")
        self.graph=graph

        return graph





if __name__ == '__main__':
    a=Graph(5,2)

    print(a.graph)
    # a.random_painting(5)


    # print(check(graph_generation(3, 33)))
