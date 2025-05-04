import open3d as o3d
import numpy as np

def count_points(points_3d):
    return points_3d.shape[1]

def compute_bounding_box(points_3d):
    min_bound = np.min(points_3d, axis=1)
    max_bound = np.max(points_3d, axis=1)
    size = max_bound - min_bound
    return min_bound, max_bound, size

def estimate_point_density(points_3d, k=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []

    for i in range(len(points_3d.T)):
        [_, idx, dists] = tree.search_knn_vector_3d(pcd.points[i], k)
        if len(dists) > 1:
            avg_dist = np.sqrt(np.mean(dists[1:]))
        else:
            avg_dist = 0
        densities.append(avg_dist)

    mean_density = np.mean(densities)
    return mean_density

def analyze_point_cloud(points_3d):
    num_points = count_points(points_3d)
    min_bound, max_bound, size = compute_bounding_box(points_3d)
    volume = np.prod(size)
    density = estimate_point_density(points_3d)
    quality = "Хорошее" if num_points > 500 and density < 0.01 else "Среднее/Плохое"

    return {
        "points": num_points,
        "bbox_size": size,
        "volume": volume,
        "avg_density": density,
        "quality": quality
    }
