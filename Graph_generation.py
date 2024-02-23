import numpy as np



class Graph:

    def random_painting(self,n_colors):
        painting={i: np.random.randint(1,n_colors)   for i in self.graph.keys()}
        print(painting)

    def graph_generation(self,n_vertice, mean_degree):
        rs = np.random.RandomState(42)
        graph = {i: {} for i in range(n_vertice)}

        n_edges = mean_degree * n_vertice / 2
        while n_edges >= 0:

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
    a=Graph()
    a.graph_generation(9,3)
    print(a.graph)
    a.random_painting(5)


    # print(check(graph_generation(3, 33)))
