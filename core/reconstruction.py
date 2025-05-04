import os
import cv2
import numpy as np
import open3d as o3d

from legacy import processor, structure, features
from legacy.camera import Camera
from legacy.processor import cart2hom
from legacy.features import find_correspondence_points
from legacy.structure import reconstruct_points


def visualize_point_cloud(points_3d):
    print("[INFO] Отображаем точечное облако")
    if points_3d.shape[0] == 4:
        points_3d = points_3d[:3, :] / points_3d[3, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    o3d.visualization.draw_geometries([pcd], window_name="Sparse Point Cloud")


def run_sparse_reconstruction(image_dir):
    print(f"[INFO] Загружаем изображения из: {image_dir}")
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm', '.bmp', '.tif', '.tiff'))
    ])
    if len(image_files) < 2:
        print("[ERROR] Нужно как минимум два изображения для реконструкции")
        return

    img1 = cv2.imread(os.path.join(image_dir, image_files[0]))
    img2 = cv2.imread(os.path.join(image_dir, image_files[1]))
    if img1 is None or img2 is None:
        print("[ERROR] Не удалось загрузить изображения")
        return

    pts1, pts2 = find_correspondence_points(img1, img2)
    print(f"[INFO] Найдено соответствий: {pts1.shape[1]}")

    K = np.array([[1000, 0, img1.shape[1] / 2], [0, 1000, img1.shape[0] / 2], [0, 0, 1]])
    cam1 = Camera(K=K, R=np.eye(3), t=np.zeros((3, 1)))
    cam2 = Camera(K=K, R=np.eye(3), t=np.array([[1], [0], [0]]))

    pts1_h = cart2hom(pts1)
    pts2_h = cart2hom(pts2)

    points_3d = reconstruct_points(pts1_h, pts2_h, cam1.P, cam2.P)
    print(f"[INFO] Реконструировано 3D-точек: {points_3d.shape[1]}")
    visualize_point_cloud(points_3d)
    return points_3d


import os
import cv2
import numpy as np

def cart2hom(points):
    return np.vstack((points, np.ones((1, points.shape[1]))))

def run_dino_reconstruction(image_dir, return_images=False):
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm', '.tif', '.tiff'))
    ])
    if len(image_files) < 2:
        raise FileNotFoundError("❌ В папке должно быть минимум 2 изображения.")

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_pair = None
    best_matches = []
    best_kp1, best_kp2 = None, None
    best_img1, best_img2 = None, None

    for i in range(len(image_files) - 1):
        path1 = os.path.join(image_dir, image_files[i])
        path2 = os.path.join(image_dir, image_files[i + 1])
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        if img1 is None or img2 is None:
            continue

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > len(best_matches):
            best_matches = matches
            best_kp1, best_kp2 = kp1, kp2
            best_img1, best_img2 = img1, img2
            best_names = (image_files[i], image_files[i + 1])

    if not best_matches:
        raise RuntimeError("❌ Не удалось найти совпадающие изображения.")

    print(f"[INFO] Лучшая пара: {best_names[0]} и {best_names[1]} ({len(best_matches)} совпадений)")

    pts1 = np.array([best_kp1[m.queryIdx].pt for m in best_matches]).T
    pts2 = np.array([best_kp2[m.trainIdx].pt for m in best_matches]).T

    points1 = cart2hom(pts1)
    points2 = cart2hom(pts2)

    # Фиктивная 3D реконструкция (временно, без E и P матриц)
    # Просто объединяем координаты для теста
    points3D = np.vstack([
        (pts1 + pts2) / 2,
        np.random.rand(1, pts1.shape[1]) * 100  # z-координата
    ])

    if return_images:
        return points3D, best_img1, best_img2, pts1, pts2, best_names

    return points3D

def export_mesh_model(self):
    if self.points_3d is None:
        self.label.setText("Нет данных для экспорта ❌")
        return

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_3d.T)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # Удаление выбросов (по желанию)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        radii = [0.005, 0.01, 0.02]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        # STL экспорт
        stl_path, _ = QFileDialog.getSaveFileName(self, "Сохранить STL", "", "STL (*.stl)")
        if stl_path:
            o3d.io.write_triangle_mesh(stl_path, mesh)
            self.label.setText(f"Сохранено в STL: {os.path.basename(stl_path)} ✅")

        # WRL экспорт
        wrl_path, _ = QFileDialog.getSaveFileName(self, "Сохранить WRL", "", "WRL (*.wrl)")
        if wrl_path:
            o3d.io.write_triangle_mesh(wrl_path, mesh)
            self.label.setText(f"Сохранено в WRL: {os.path.basename(wrl_path)} ✅")

    except Exception as e:
        self.label.setText("Ошибка экспорта в STL/WRL ❌")
        print(f"[ERROR] {e}")
