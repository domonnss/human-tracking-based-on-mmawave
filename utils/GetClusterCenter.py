import numpy as np


def get_cluster_centers(cluster_centers, point):
    center = np.mean(point, axis=0)
    cluster_centers.append(center)
    return np.array(cluster_centers)
