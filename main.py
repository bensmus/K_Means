import numpy as np
import random
import matplotlib.pyplot as plt

# K-Means clustering algorithm
# https://stanford.edu/~cpiech/cs221/handouts/kmeans.html

N = 30  # number of points 
K = 2  # number of clusters
red = '#fc0303'
green = '#0bfc03'
blue = '#030bfc'
coloring = [red, blue]  # because we have two different color clusters
count = 10  # number of times we get new centroids and associate clusters

points = np.random.rand(N, 2)
'''
[
    [88, 2],
    [7, 27]
    [35, 6]...
]
'''
'''
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y)
'''
def plot_points(points, color):
    '''Arg points is a mx2 array, color is hex string'''
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, c=color)

''' WORKS
plot_points(points[:4], red)
plot_points(points[4:], blue)
plt.show()
'''


def centroids_init(points, K):
    '''Choose K random points to be the initial centroids (around which to cluster).'''
    
    centroids = np.array([random.choice(points) for _ in range(K)])
    return centroids  # list with K points


centroids = centroids_init(points, K)


for _ in range(count):
    clusters = get_clusters(centroids, points)
    centroids = get_centroids(clusters)

# we are done the algorithm
for index, cluster in enumerate(clusters):
    plot_points(cluster, coloring[index])
plt.show()
