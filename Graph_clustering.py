import sys

import numpy
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from IPython.display import SVG, display

from scipy import sparse
from sknetwork.clustering import Louvain, Leiden, KCenters, PropagationClustering
from sknetwork.data import from_edge_list, from_adjacency_list, from_graphml, from_csv
from sknetwork.visualization import visualize_graph, visualize_bigraph

from Graph_generation import Graph
from Graph_visualization import show_graph


def clustering(graph: Graph,n_clusters: int,edge_tr=None):
    # graph.graph_generation_knn(graph.n_vertices,graph.mean_degree)
    ndpos=np.zeros(len(graph.pos))
    adjency=None
    if edge_tr is not None:
        adjacency = graph.to_adjacency_matrix(edge_tr)
    else:
        adjacency = graph.to_adjacency_matrix()
    adjacency = sparse.csr_matrix(adjacency)
    kcenters = KCenters(n_clusters=n_clusters)
    kcenters.fit(adjacency)
    labels = kcenters.fit_predict(adjacency)
    return labels
    # show_graph(graph,graph.coloring,clustering=labels)

# def decomposition(graph,labels):



if __name__ == '__main__':

    g= Graph()
    g.init_by_graph([])
    # g.init_by_n_v_mean_degree( 75, 15)
    # labels=clustering(g,2)
    #
    # # print(g.split_graph_by_clusters(labels))
    # show_graph(g, g.coloring, clustering=labels)

def display_svg_in_webview(svg_string):
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout()

    webview = QWebEngineView()
    webview.setHtml(svg_string)  # Pass SVG string directly as HTML
    layout.addWidget(webview)

    window.setLayout(layout)
    window.setWindowTitle("SVG in WebView")
    window.show()

    sys.exit(app.exec_())
