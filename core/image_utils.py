# === core/image_utils.py ===
import cv2

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)

# === core/mesh_utils.py (добавим экспорт WRL) ===
import open3d as o3d

def export_wrl(points_3d, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)
    pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True, write_vertex_normals=True)

# === gui/photo_editor_window.py ===
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QSlider, QHBoxLayout, QFileDialog, QScrollArea, QWidget, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import os
from core.image_utils import adjust_brightness_contrast

class PhotoEditorWindow(QDialog):
    def __init__(self, image_paths, used_paths):
        super().__init__()
        self.setWindowTitle("Редактор фотографий")
        self.setMinimumSize(800, 600)
        self.image_paths = image_paths
        self.used_paths = used_paths
        self.layout = QVBoxLayout(self)

        scroll = QScrollArea()
        content = QWidget()
        grid = QGridLayout(content)

        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(150, 100, Qt.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(pix)
            if path not in used_paths:
                label.setStyleSheet("border: 2px solid red;")
            else:
                label.setStyleSheet("border: 1px solid green;")

            label.mousePressEvent = lambda _, p=path: self.edit_image(p)
            grid.addWidget(label, i // 4, i % 4)

        scroll.setWidget(content)
        self.layout.addWidget(scroll)

    def edit_image(self, path):
        editor = ImageAdjustDialog(path)
        editor.exec_()

class ImageAdjustDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Коррекция изображения")
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.current = self.original.copy()

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.update_display()

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.valueChanged.connect(self.apply_adjustments)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.valueChanged.connect(self.apply_adjustments)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(QLabel("Яркость"))
        self.layout.addWidget(self.brightness_slider)
        self.layout.addWidget(QLabel("Контраст"))
        self.layout.addWidget(self.contrast_slider)
        self.save_button = QPushButton("Сохранить во временную память")
        self.save_button.clicked.connect(self.accept)
        self.layout.addWidget(self.save_button)

    def update_display(self):
        img_rgb = cv2.cvtColor(self.current, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(300, 200, Qt.KeepAspectRatio))

    def apply_adjustments(self):
        bright = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        self.current = adjust_brightness_contrast(self.original, bright, contrast)
        self.update_display()

# === main_window.py (фрагмент, добавляем вызов) ===
from gui.photo_editor_window import PhotoEditorWindow

def open_photo_editor(self):
    image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('jpg', 'png', 'ppm'))]
    used_files = getattr(self, 'used_image_paths', [])  # список изображений, использованных в реконструкции
    editor = PhotoEditorWindow(image_files, used_files)
    editor.exec_()

# === main_window.py (в run_reconstruction — сохраняем used_image_paths и время) ===
start_time = time.time()
...
self.used_image_paths = [os.path.join(self.image_dir, best_names[0]), os.path.join(self.image_dir, best_names[1])]
...
elapsed = time.time() - start_time
self.stats_label.setText(self.stats_label.text() + f"\n📸 Использовано фото: 2 из {len(os.listdir(self.image_dir))}"
                                             f"\n⏱️ Время реконструкции: {elapsed:.2f} сек")
