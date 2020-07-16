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


def plot_points(points, color):
    '''Arg points is a mx2 array, color is hex string'''
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, c=color)


def centroids_init(points, K):
    '''Choose K random points to be the initial centroids (around which to cluster).'''
    
    centroids = [list(random.choice(points)) for _ in range(K)]
    return centroids  # list with K points


def find_closest_centroid(centroids, point):
    '''returns [x, y]: the coors of closest centroid in centroids'''

    distances = []
    for centroid in centroids:
        distances.append(np.linalg.norm(np.array(centroid) - np.array(point)))
    min_index = distances.index(min(distances))
    
    return centroids[min_index]


def get_clusters(centroids, points):
    '''returns an array of clusters (3D array)'''
    
    clusters = [[] for _ in centroids]
    for point in points:
        print(centroids, clusters)
        # deciding which cluster to assign to 
        centroid = find_closest_centroid(centroids, point)
        cluster = clusters[centroids.index(centroid)]
        cluster.append(point)
    
    return clusters

    
centroids = centroids_init(points, K)
clusters = get_clusters(centroids, points)
for i in range(K):
    plot_points(np.array(clusters[i]), coloring[i])
plt.show()
'''
for _ in range(count):
    clusters = get_clusters(centroids, points)
    centroids = get_centroids(clusters)

# we are done the algorithm
for i in range(K):
    plot_points(clusters[i], coloring[i])
plt.show()
'''