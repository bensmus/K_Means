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
    
    centroids = np.array([random.choice(points) for _ in range(K)])
    return centroids  # np.array with K points


def find_closest_centroid(centroids, point):
    '''returns [x, y]: the coors of closest centroid in centroids'''

    distances = []
    for centroid in centroids:
        distances.append(np.linalg.norm(centroid - point))
    min_index = distances.index(min(distances))
    
    return centroids[min_index]


def get_clusters(centroids, points):
    '''returns an dict with values being clusters (2D numpy arrays)'''
    
    cluster_dict = {i:np.zeros((1, 2)) for i in range(K)}

    for point in points:
        
        # deciding which cluster to assign to 
        centroid = find_closest_centroid(centroids, point)
        i = centroids.tolist().index(centroid.tolist())  # use the list.index(element) method
        
        comparison = cluster_dict[i] == np.zeros((1, 2))
        equal_arrays = comparison.all()  # boolean values
        if equal_arrays:
            cluster_dict[i] = point.reshape(1, 2)  # making it 2D
        else:
            cluster_dict[i] = np.append(cluster_dict[i], point.reshape(1, 2), axis=0)

    return cluster_dict
    

def get_centroids(clusters):
    '''returns an array of centroids (2D array)'''
    
    centroids = np.zeros((K, 2))
    for i, cluster in enumerate(clusters.values()):
        centroid = (np.cumsum(cluster, axis=0)[-1]) / cluster.shape[0]
        centroids[i] = centroid
    
    return np.array(centroids)


centroids = centroids_init(points, K)

for _ in range(count):
    clusters = get_clusters(centroids, points)
    centroids = get_centroids(clusters)

# we are done the algorithm
for i in range(K):
    plot_points(clusters[i], coloring[i])
plot_points(centroids, green)
plt.show()
