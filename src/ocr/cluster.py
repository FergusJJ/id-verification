import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def get_center_coordinates(box_coordinates):
    structured_data = []
    for (x1, y1), (x2, y2) in box_coordinates:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        structured_data.append(
            {"center": (int(center_x), int(center_y)), "box": ((x1, y1), (y1, y2))}
        )
    return structured_data


def run_clustering(box_coordinates):
    structured_data = get_center_coordinates(box_coordinates)
    # Extract just the center coordinates for clustering
    centers = np.array([data["center"] for data in structured_data])
    return centers
    print(centers)
    clustering = DBSCAN(eps=5, min_samples=1).fit(centers)
    # print(clustering.labels_)
    # Create a dictionary to hold the boxes for each cluster
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(structured_data[i]["box"])
    return clusters


"""
    clustering = DBSCAN(eps=5, min_samples=1).fit(centers)
    clusters = defaultdict(list)
    for idx, label in enumerate(clustering.labels_):
        clusters[label].append(structured_data[idx]["box"])
"""
