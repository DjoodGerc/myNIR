# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    nbrs.kneighbors(X)
    xx=[X[i,0] for i in range(len(X))]
    xy=[X[i,1] for i in range(len(X))]
    labels=[i for i in range(len(X))]
    plt.scatter(xx,xy)

    for x_coord, y_coord, label in zip(xx, xy, labels):
        plt.text(x_coord, y_coord, label)
    distances, indices = nbrs.kneighbors(X)
    print(distances)
    print(indices)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
