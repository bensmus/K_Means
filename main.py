import numpy as np
import random
import sys
import pygame
import pygame.freetype
import colorsys
#pylint: disable=no-member
pygame.init()


def spread_colors(count):
    '''
    Count is the amount of random colors we want to generate.
    Colorsys gives rgb as float between 0 and 1. Pygame supports 8 bit color (255^3) possibilities.
    Generate a 2D list of spaced RGB colors.
    '''
    change = 1/count  # if we have less groups, we need more different hues
    huelist = [(i * change) for i in range(count)]
    rgb = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in huelist]  

    for i, color in enumerate(rgb):
        rgb[i] = list(map(lambda x: int(x*255), color))

    return rgb


size = 640, 480
screen = pygame.display.set_mode(size)
overlay = screen.copy()
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Comic Sans MS', 24)

white = 255, 255, 255
black = 0, 0, 0

# K-Means clustering algorithm
# https://stanford.edu/~cpiech/cs221/handouts/kmeans.html

N = int(sys.argv[1])  # number of points as a command line argument
K = int(sys.argv[2])  # number of clusters as a command line argument
colors = spread_colors(K + 1)  # different colors, K for clusters, first one for centroids 
centroid_color = colors[0]
clusters_colors = colors[1:]

points = np.random.rand(N, 2)
# we need to get these to a pixelable value
points *= 400  # max pixel value 400
points += 10  # getting a safe margin
    

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


def draw_clusters_and_centroids(clusters, centroids, clusters_colors, centroid_color):
    '''Draws points to screen. Expects iterable for clusters_colors.'''

    for i, cluster in enumerate(clusters.values()):
        for point in cluster:
            pygame.draw.circle(screen, clusters_colors[i], list(map(int, list(point))), 5)
    
    for centroid in centroids:
        pygame.draw.circle(screen, centroid_color, list(map(int, list(centroid))), 5)


centroids = np.random.rand(K, 2)
print(centroids)

time = 0
space_count = 0
update_time = 1000
paused = False  # allowing pause
while True:
    dt = clock.tick(60)

    if not paused:
        time += dt
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_count += 1
   
    if space_count % 2 == 1:
        paused = True
    else:
        paused = False
    
    if time > update_time:  # every 1000 milliseconds
        update_time += 1000
        screen.fill(white)
        
        # K Means!
        clusters = get_clusters(centroids, points)
        centroids = get_centroids(clusters)
        draw_clusters_and_centroids(clusters, centroids, clusters_colors, centroid_color)     
        pygame.display.flip()


