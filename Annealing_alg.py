import numpy as np

from Graph_generation import Graph


def annealing_algorithm(g:Graph,n_colors:int,t: float,rate_of_inrease)->dict:
    """
    Метод Отжига
    """
    g.coloring={i:np.random.randint(0,n_colors) for i in g.graph.keys()}
    rs = np.random

    #условие остановки

    while 1/t>10**-10:

        # Берем случайную вершину, берем случайный цвет оценивам разницу между ошибками со старым и новым цветом

        vert = np.random.randint(0, len(g.graph.keys()))
        new_color = (rs.randint(1, n_colors) + g.coloring[vert]) % n_colors
        new_coloring = g.coloring.copy()
        new_coloring[vert] = new_color

        delta = g.get_d_obj(vert,new_color)[0]
        print(delta)
        print(1/t)
        r = np.random.rand()

        # если раскраска улучшилась : принимаем; нет: c вероятностью np.exp(-t * delta) принимаем

        if delta <= 0:
            g.coloring = new_coloring

        elif r < np.exp(-t * delta):
            
            g.coloring = new_coloring


        t *= rate_of_inrease
        
    return g.coloring

