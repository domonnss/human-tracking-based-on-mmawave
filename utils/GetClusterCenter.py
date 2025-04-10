import numpy as np


def get_cluster_centers(cluster_centers_dict, point, label):
    center = np.mean(point, axis=0)
    if label not in cluster_centers_dict:
        cluster_centers_dict[label] = []
    cluster_centers_dict[label].append(center)
    return cluster_centers_dict
