from Graph_generation import Graph
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def errors_convergence(n_vertice,mean_degree, n_color,t,dec, seed=42):
    g = Graph(n_vertice, mean_degree)


    fst_ann = g.annealing(n_color,t, dec)

    coloring_hc = g.hill_climbing(n_color)

    snd_ann=g.annealing_snd(n_color,t, dec)

    plt.plot([i for i in range(1,len(fst_ann[1])+1)],fst_ann[1])
    plt.legend("первый отжиг (авторский)")
    plt.plot([i for i in range(1, len(snd_ann[1]) + 1)], snd_ann[1])
    plt.legend("второй отжиг ")
    plt.plot([i for i in range(1, len(coloring_hc[1]) + 1)], coloring_hc[1])
    plt.legend("восхождение на гору")
    print(g.get_obj(g.random_coloring(n_color)))
    print()
    print(g.get_obj(fst_ann[0]))
    print(g.get_obj(snd_ann[0]))
    print(g.get_obj(coloring_hc[0]))
    print()
    print(fst_ann[1][len(fst_ann[1]) - 1])
    print(snd_ann[1][len(snd_ann[1]) - 1])
    print(coloring_hc[1][len(coloring_hc[1]) - 1])

    plt.xlabel("итерация")
    plt.ylabel("ошибка")
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
   # errors_convergence(500,20,10,12,1.001)
   errors_convergence(17, 7,4, 12, 1.01)


