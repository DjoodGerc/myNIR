import time

from tqdm import trange

from Graph_generation import Graph
import matplotlib

from Gurobi_standart import gurobi_whith_clustering
from g_coloring import g_one, g_two, g_three, g_four

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plots(bot, top, n_colors):
    hc_obj = []

    ann_obj = []
    gurobi_obj = []
    g_one_obj = []
    g_two_obj = []
    g_three_obj = []
    g_four_obj = []

    hc_time = []
    ann_time = []
    gurobi_time = []
    g_one_time = []
    g_two_time = []
    g_three_time = []
    g_four_time = []
    x = []
    for i in trange(bot, top,5):
        g = Graph()
        g.graph_generation_knn(i, 15)

        t1 = time.time()
        hc_obj.append(g.get_obj(g.hill_climbing(n_colors)[0]))
        t2 = time.time()
        hc_time.append(t2 - t1)

        t1 = time.time()
        ann_obj.append(g.get_obj(g.annealing(n_colors, 8.5, 1.0004)[0]))
        t2 = time.time()
        ann_time.append(t2 - t1)

        # t1 = time.time()
        # gurobi_obj.append(gurobi_whith_clustering(g, n_colors)[0])
        # t2 = time.time()
        # gurobi_time.append(t2 - t1)

        t1 = time.time()
        g_one_obj.append(g_one(g)[1])
        t2 = time.time()
        g_one_time.append(t2 - t1)

        t1 = time.time()
        g_two_obj.append(g_two(g, n_colors)[1])
        t2 = time.time()
        g_two_time.append(t2 - t1)

        t1 = time.time()
        g_three_obj.append(g_three(g, n_colors)[1])
        t2 = time.time()
        g_three_time.append(t2 - t1)

        t1 = time.time()
        g_four_obj.append(g_four(g, n_colors)[1])
        t2 = time.time()
        g_four_time.append(t2 - t1)

        # my_ann_obj.append(g.get_obj(g.annealing_snd(n_colors, 3, 1.005)[0]))
        x.append(i)

    my_file = open("otus.txt", "w")
    filename = f"{bot} - {top}, {n_colors}, 15 mean deg, 1.3 start temp, 1.006 - v"
    file_write(filename, gurobi_obj, hc_obj, ann_obj, g_one_obj, g_two_obj, g_three_obj, g_four_obj)

    plt.subplot(1, 2, 1)
    # plt.plot(x, gurobi_obj, "red")
    plt.plot(x, hc_obj, "blue")
    plt.plot(x, ann_obj, "black")
    plt.plot(x, g_one_obj, "magenta")
    plt.plot(x, g_two_obj, "yellow")
    plt.plot(x, g_three_obj, "orange")
    plt.plot(x, g_four_obj, "green")
    plt.legend(['Восхождение на гору', 'Отжиг', 'G1', 'G2', 'G3', 'G4'])
    # plt.legend(['Эталон', 'Восхождение на гору', 'Отжиг', 'G1', 'G2', 'G3', 'G4'])
    plt.title('Oшибка')
    plt.xlabel('кол-во вершин')
    plt.ylabel('сумма весов ребер, инцидентных вершинам одного цвета (ошибка)')

    plt.subplot(1, 2, 2)
    # plt.plot(x, gurobi_time, "red")
    plt.plot(x, hc_time, "blue")
    plt.plot(x, ann_time, "black")
    plt.plot(x, g_one_time, "magenta")
    plt.plot(x, g_two_time, "yellow")
    plt.plot(x, g_three_time, "orange")
    plt.plot(x, g_four_time, "green")
    plt.legend(['Восхождение на гору', 'Отжиг', 'G1', 'G2', 'G3', 'G4'])
    # plt.legend(['Эталон', 'Восхождение на гору', 'Отжиг', 'G1', 'G2', 'G3', 'G4'])
    plt.title('время выполнения')
    plt.xlabel('кол-во вершин')
    plt.ylabel('Время выполнения алгоритма')

    plt.show()


def errors_convergence(n_vertice, mean_degree, n_color, t, dec, seed=42):
    g = Graph(n_vertice, mean_degree)
    graphG = g.graph
    coloringG = g.random_coloring(n_color)

    fst_ann = g.annealing(n_color, t, dec, coloring=coloringG.copy())

    coloring_hc = g.hill_climbing(n_color, coloring=coloringG.copy())

    snd_ann = g.annealing_snd(n_color, t, dec, coloring=coloringG.copy())

    plt.plot([i for i in range(1, len(fst_ann[1]) + 1)], fst_ann[1], "blue")
    plt.legend("fst ann")
    plt.plot([i for i in range(1, len(snd_ann[1]) + 1)], snd_ann[1], "red")
    plt.legend("snd ann")
    plt.plot([i for i in range(1, len(coloring_hc[1]) + 1)], coloring_hc[1], color="black")
    plt.legend("hill climbing")

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


def file_write(filename, et, hc, ann, g1, g2, g3, g4):
    file = open(f"{filename}.txt", 'w')
    print(et, file=file, sep=" ")
    file.write("\n")
    print(hc, file=file, sep=" ")
    file.write("\n")
    print(ann, file=file, sep=" ")
    file.write("\n")
    print(g1, file=file, sep=" ")
    file.write("\n")
    print(g2, file=file, sep=" ")
    file.write("\n")
    print(g3, file=file, sep=" ")
    file.write("\n")
    print(g4, file=file, sep=" ")
    file.write("\n")
    file.close()


if __name__ == '__main__':
    plots(40, 1000, 5)
# errors_convergence(500,15,7,10,1.001)
# errors_convergence(17, 7,4, 12, 1.01)
