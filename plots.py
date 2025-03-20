from Graph_generation import Graph
import matplotlib

from Gurobi_standart import gurobi_whith_clustering

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plots(bot,top,n_colors):

    hc_obj=[]
    ann_obj=[]
    gurobi_obj=[]
    x=[]
    for i in range(bot,top,20):
        g=Graph()
        g.graph_generation_knn(i,20)
        hc_obj.append(g.get_obj(g.hill_climbing(n_colors)[0]))
        ann_obj.append(g.get_obj(g.annealing(n_colors,10,1.001)[0]))
        gurobi_obj.append(gurobi_whith_clustering(g,n_colors)[0])
        x.append(i)
    plt.plot(x,gurobi_obj,"red")
    plt.plot(x,hc_obj,"blue")
    plt.plot(x,ann_obj,"black")
    plt.show()
def errors_convergence(n_vertice,mean_degree, n_color,t,dec, seed=42):
    g = Graph(n_vertice, mean_degree)
    graphG=g.graph
    coloringG=g.random_coloring(n_color)


    fst_ann = g.annealing(n_color,t, dec,coloring=coloringG.copy())

    coloring_hc = g.hill_climbing(n_color,coloring=coloringG.copy())

    snd_ann=g.annealing_snd(n_color,t, dec,coloring=coloringG.copy())




    plt.plot([i for i in range(1,len(fst_ann[1])+1)],fst_ann[1],"blue")
    plt.legend("fst ann")
    plt.plot([i for i in range(1, len(snd_ann[1]) + 1)], snd_ann[1],"red")
    plt.legend("snd ann")
    plt.plot([i for i in range(1, len(coloring_hc[1]) + 1)], coloring_hc[1],color="black")
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


if __name__ == '__main__':
    plots(40,250,6)
   # errors_convergence(500,15,7,10,1.001)
   # errors_convergence(17, 7,4, 12, 1.01)


