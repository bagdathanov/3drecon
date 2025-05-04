import open3d as o3d
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog

def build_mesh_from_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """
    Строит треугольную сетку из облака точек методом Ball Pivoting.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    radii = [0.005, 0.01, 0.02]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh

def export_mesh_model(points_3d: np.ndarray, filepath: str):
    """
    Экспортирует mesh из точек в STL-файл.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    mesh = build_mesh_from_point_cloud(pcd)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".stl":
        o3d.io.write_triangle_mesh(filepath, mesh)
    else:
        raise ValueError(f"Unsupported mesh export format: {ext}")

import time
def visualize_quality_progression(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)

    # 1. Облако точек
    o3d.visualization.draw_geometries([pcd], window_name="Этап 1: Облако точек", width=800, height=600)
    time.sleep(1)

    # 2. Упрощённый меш (alpha shape)
    mesh_simple = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.1)
    mesh_simple.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_simple], window_name="Этап 2: Грубая триангуляция", width=800, height=600)
    time.sleep(1)

    # ✅ Важно: оценка нормалей перед Poisson
    pcd.estimate_normals()

    # 3. Плотный меш (Poisson)
    mesh_detailed, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh_detailed.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_detailed], window_name="Этап 3: Плотный меш", width=800, height=600)


def generate_mesh_stages(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    pcd.estimate_normals()

    # Разреженная модель (low detail)
    mesh_simple = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.1)

    # Базовая модель (average detail)
    mesh_mid, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)

    # Детализированная модель
    mesh_high, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    return [mesh_simple, mesh_mid, mesh_high]
def animated_quality_progression(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    pcd.estimate_normals()

    # Простая меш-сетка
    mesh_simple = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.1)
    mesh_simple.compute_vertex_normals()

    # Плотная Poisson-сетка
    mesh_detailed, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh_detailed.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Анимация оценки качества", width=960, height=720)

    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.5)

    vis.clear_geometries()
    vis.add_geometry(mesh_simple)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.5)

    vis.clear_geometries()
    vis.add_geometry(mesh_detailed)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(2)

    vis.destroy_window()


def build_mesh(pcd: o3d.geometry.PointCloud, coarse=True):
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    radii = [0.02, 0.04] if coarse else [0.005, 0.01, 0.02]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh
