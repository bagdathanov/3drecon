import open3d as o3d
import numpy as np
import os
import cv2
from PyQt5.QtWidgets import QFileDialog

def save_ply(filepath, points_3d, image=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    if image is not None:
        height, width, _ = image.shape
        colors = []
        for x, y, z in points_3d.T:
            u = int(x * width / 2 + width / 2)
            v = int(-y * height / 2 + height / 2)
            if 0 <= u < width and 0 <= v < height:
                color = image[v, u] / 255.0
            else:
                color = [0.5, 0.5, 0.5]
            colors.append(color[::-1])
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    o3d.io.write_point_cloud(filepath, pcd)

def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd, np.asarray(pcd.points).T

def export_obj(filepath, points_3d, image):
    mtl_path = os.path.splitext(filepath)[0] + ".mtl"
    tex_path = os.path.splitext(filepath)[0] + "_texture.png"
    with open(filepath, 'w') as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\nusemtl material_0\n")
        height, width, _ = image.shape
        tex_coords = []
        for pt in points_3d.T:
            x, y, z = pt
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            tex_coords.append((0.5 + x / 2, 0.5 - y / 2))
        for u, v in tex_coords:
            f.write(f"vt {u:.4f} {v:.4f}\n")
        for i in range(len(points_3d.T)):
            f.write(f"f {i + 1}/{i + 1}\n")
    with open(mtl_path, 'w') as f:
        f.write("newmtl material_0\nKa 1 1 1\nKd 1 1 1\nmap_Kd " + os.path.basename(tex_path) + "\n")
    cv2.imwrite(tex_path, image)

