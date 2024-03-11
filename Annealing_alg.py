import numpy as np

from Graph_generation import Graph

"""
Метод Отжига
"""
def annealing_algorithm(g:Graph,n_colors:int,t: float,rate_of_inrease)->dict:
    g.painting={i:np.random.randint(0,n_colors) for i in g.graph.keys()}
    rs = np.random
    """
    условие остановки
    """
    while 1/t>10**-50:
        """
        Берем случайную вершину, берем случайный цвет
        оценивам разницу между ошибками со старым и новым цветом
        """
        vert = np.random.randint(0, len(g.graph.keys()))
        new_color = (rs.randint(1, n_colors) + g.painting[vert]) % n_colors
        new_painting = g.painting.copy()
        new_painting[vert] = new_color

        delta = g.find_differents_painting_with_one_changed_color(vert,new_color)[0]

        r = np.random.rand()
        """
        если раскраска улучшилась : принимаем
        нет: c вероятностью np.exp(-t * delta) принимаем
        """
        if delta <= 0:
            g.painting = new_painting

        elif r < np.exp(-t * delta):
            # print(f"rand={r}")
            g.painting = new_painting


        t *= rate_of_inrease
        
    return g.painting
