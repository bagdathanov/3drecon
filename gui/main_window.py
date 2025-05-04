from PyQt5.QtWidgets import QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from gui.vispy_widget import Vispy3DWidget
from core.reconstruction import run_dino_reconstruction
from PyQt5.QtCore import Qt
import os
import cv2
import open3d as o3d
import trimesh

import multiprocessing
from core.io_utils import save_ply, load_ply, export_obj
from core.mesh_utils import export_mesh_model
from core.stats import analyze_point_cloud
from core.mesh_utils import visualize_quality_progression, generate_mesh_stages
import numpy as np
from core.mesh_utils import animated_quality_progression
from vispy.scene import visuals
from vispy.color import Color
from vispy.visuals.transforms import MatrixTransform
import time
from core.mesh_utils import generate_mesh_stages
from vispy.scene import visuals
import numpy as np
import time
from PyQt5.QtCore import QTimer
from gui.photo_review_window import PhotoReviewWindow


def open3d_viewer(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    o3d.visualization.draw_geometries([pcd], window_name="Open3D Viewer")


class MainWindow(QMainWindow):
    def __init__(self, user_id=None, user_role="user"):
        super().__init__()
        self.setWindowTitle("3D Reconstruction")
        self.setGeometry(100, 100, 1200, 700)
        self.setAcceptDrops(True)

        self.user_id = user_id
        self.user_role = user_role
        self.is_admin = (self.user_role == "admin")

        self.label = QLabel("Выберите папку с изображениями", alignment=Qt.AlignCenter)
        self.btn_load = QPushButton("Загрузить изображения")
        self.btn_load.clicked.connect(self.load_images)

        self.btn_logout = QPushButton("Выйти из аккаунта")
        self.btn_logout.clicked.connect(self.logout)

        self.btn_run = QPushButton("Запустить реконструкцию")
        self.btn_run.clicked.connect(self.run_reconstruction)
        self.btn_run.setEnabled(False)

        self.stats_label = QLabel("Информация о модели появится здесь.")
        self.stats_label.setAlignment(Qt.AlignLeft)
        self.stats_label.setWordWrap(True)

        self.btn_import_model = QPushButton("Импортировать модель")
        self.btn_import_model.clicked.connect(self.import_model_dialog)

        self.btn_open3d = QPushButton("Открыть в Open3D")
        self.btn_open3d.clicked.connect(self.show_open3d)
        self.btn_open3d.setEnabled(False)

        self.btn_save = QPushButton("Экспортировать модель")
        self.btn_save.clicked.connect(self.export_model_dialog)
        self.btn_save.setEnabled(False)

        self.fig = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.fig)

        self.vispy_widget = Vispy3DWidget()

        self.btn_review_photos = QPushButton("Обзор и коррекция фото")
        self.btn_review_photos.clicked.connect(self.open_photo_review)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_review_photos)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_open3d)
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_import_model)
        layout.addWidget(self.btn_logout)

        layout.addWidget(self.vispy_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_dir = None
        self.points_3d = None
        self.pcd = None

        self.restrict_ui_by_role()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.toLocalFile().endswith(".ply"):
                self.load_ply_file(url.toLocalFile())

    def load_images(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if path:
            self.image_dir = path
            self.label.setText(f"Выбрана папка: {os.path.basename(path)}")
            self.btn_run.setEnabled(True)

    def import_model_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Импортировать 3D-модель", "", "3D Models (*.ply *.stl *.wrl)"
        )
        if not filepath:
            return

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".ply":
                self.pcd = o3d.io.read_point_cloud(filepath)
            elif ext in [".stl", ".wrl"]:
                mesh = o3d.io.read_triangle_mesh(filepath)
                self.pcd = mesh.sample_points_uniformly(number_of_points=50000)
            else:
                self.label.setText("Неподдерживаемый формат ❌")
                return

            self.points_3d = np.asarray(self.pcd.points).T
            self.fig.clf()
            ax = self.fig.add_subplot(111, projection='3d')
            ax.scatter(self.points_3d[0], self.points_3d[1], self.points_3d[2], c='black', s=1)
            ax.set_title(f"Импортировано: {os.path.basename(filepath)}")
            self.canvas.draw()
            self.vispy_widget.set_point_cloud(self.points_3d)
            self.label.setText(f"Импорт завершён ✅ ({os.path.basename(filepath)})")
            self.update_export_buttons()

        except Exception as e:
            self.label.setText("Ошибка при импорте ❌")
            print(f"[ERROR] {e}")


    def export_model_dialog(self):
        if self.points_3d is None:
            self.label.setText("Нет данных для экспорта ❌")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Экспортировать 3D-модель", "",
            "Все форматы (*.ply *.obj *.stl *.wrl);;PLY (*.ply);;OBJ (*.obj);;STL (*.stl);;WRL (*.wrl)"
        )
        if not filepath:
            return

        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".ply":
                save_ply(filepath, self.points_3d, self.img1 if hasattr(self, 'img1') else None)

            elif ext == ".obj":
                export_obj(filepath, self.points_3d, self.img1 if hasattr(self, 'img1') else None)

            elif ext == ".stl":
                export_mesh_model(self.points_3d, filepath)

            elif ext == ".wrl":
                # Сначала сохраняем как временный .ply
                temp_ply = filepath.replace(".wrl", "_temp.ply")
                save_ply(temp_ply, self.points_3d, self.img1 if hasattr(self, 'img1') else None)

                # Загружаем и экспортируем как WRL через trimesh
                mesh = trimesh.load(temp_ply)
                mesh.export(filepath)

                os.remove(temp_ply)  # удалим временный файл

            else:
                self.label.setText("Неподдерживаемый формат ❌")
                return

            self.label.setText(f"Экспорт завершён: {os.path.basename(filepath)} ✅")
        except Exception as e:
            self.label.setText("Ошибка при экспорте ❌")
            print(f"[ERROR] {e}")

    def run_reconstruction(self):
        if not self.image_dir:
            return

        try:
            result = run_dino_reconstruction(self.image_dir, return_images=True)
            import time
            start_time = time.time()  # ⏱️ старт
            if result is None:
                self.label.setText("Ошибка реконструкции ❌")
                return

            points_3d, self.img1, self.img2, self.points1, self.points2, best_names = result
            elapsed_time = time.time() - start_time  # ⏱️ конец

            # Преобразование в [3, N], если в гомогенной форме
            if points_3d.shape[0] == 4:
                points_3d = points_3d[:3, :] / points_3d[3, :]

            self.points_3d = points_3d
            stats = analyze_point_cloud(points_3d)
            self.update_stats_label(stats)
            # 🔽 Сохраняем использованные и плохие фото
            self.used_image_paths = [
                os.path.join(self.image_dir, name) for name in best_names
            ]
            all_image_paths = sorted([
                os.path.join(self.image_dir, f)
                for f in os.listdir(self.image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm'))
            ])
            self.low_quality_images = [
                path for path in all_image_paths if path not in self.used_image_paths
            ]

            self.fig.clf()
            ax1 = self.fig.add_subplot(1, 3, 1)
            ax2 = self.fig.add_subplot(1, 3, 2)
            ax3 = self.fig.add_subplot(1, 3, 3, projection='3d')

            ax1.imshow(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
            ax1.plot(self.points1[0], self.points1[1], 'r.', markersize=2)
            ax1.set_title("Image 1")
            ax1.axis('off')

            ax2.imshow(cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB))
            ax2.plot(self.points2[0], self.points2[1], 'r.', markersize=2)
            ax2.set_title("Image 2")
            ax2.axis('off')

            ax3.plot(points_3d[0], points_3d[1], points_3d[2], 'k.', markersize=1)
            ax3.set_title("3D Reconstructed")
            ax3.view_init(elev=135, azim=90)

            self.canvas.draw()
            self.animate_points_progression(points_3d)

        except Exception as e:
            self.label.setText("Ошибка реконструкции ❌")
            print(f"[ERROR] {e}")

    def show_open3d(self):
        if self.points_3d is not None:
            multiprocessing.Process(target=open3d_viewer, args=(self.points_3d,)).start()
        else:
            self.label.setText("Нет 3D-точек для отображения ❌")

    def update_export_buttons(self):
        has_points = self.points_3d is not None and self.points_3d.shape[1] > 0
        self.btn_save.setEnabled(has_points)
        self.btn_open3d.setEnabled(has_points)

    def update_stats_label(self, stats: dict):
        total_photos = len([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm'))
        ])
        recon_time = getattr(self, "reconstruction_time", None)

        if recon_time is not None:
            recon_time_str = f"{recon_time:.2f} сек"
        else:
            recon_time_str = "?"

        self.stats_label.setText(
            f"📊 Статистика модели:\n"
            f"- Точек: {stats['points']}\n"
            f"- Размер bbox: {stats['bbox_size']}\n"
            f"- Объём: {stats['volume']:.4f}\n"
            f"- Плотность: {stats['avg_density']:.6f}\n"
            f"- Качество: {stats['quality']}\n"
            f"- Использовано фото: {total_photos}\n"
        )

    def restrict_ui_by_role(self):
        if not self.is_admin:
            self.btn_load.setEnabled(False)
            self.btn_run.setEnabled(False)
            self.label.setText("🔒 Вы вошли как пользователь. Доступен только просмотр загруженных моделей.")

    def logout(self):
        self.close()
        from gui.login_window import LoginWindow
        login_window = LoginWindow()
        if login_window.exec_() == login_window.Accepted:
            from gui.main_window import MainWindow
            global main_window_ref
            main_window_ref = MainWindow(user_id=login_window.user_id, user_role=login_window.user_role)
            main_window_ref.show()

    def animate_points_progression(self, full_points):
        self.points_3d = full_points.copy()
        anim_points = full_points.copy()

        if anim_points.shape[0] == 4:
            denominator = np.where(anim_points[3, :] == 0, 1e-8, anim_points[3, :])
            anim_points = anim_points[:3, :] / denominator

        centered = anim_points - np.mean(anim_points, axis=1, keepdims=True)
        scale = np.max(np.linalg.norm(centered, axis=0))
        anim_points = centered / scale

        self._full_points = anim_points
        self._animation_index = 0
        self._step_size = max(1, anim_points.shape[1] // 200)

        self._animation_start_time = time.time()  # 🔥 Запомни время старта

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_animation_frame)
        self._timer.start(50)

    def _update_animation_frame(self):
        end = self._animation_index + self._step_size
        current = self._full_points[:, :end]
        self.vispy_widget.set_point_cloud(current)
        self._animation_index = end

        if self._animation_index >= self._full_points.shape[1]:
            self._timer.stop()
            self.points_3d = self._full_points

            elapsed_anim = time.time() - self._animation_start_time  # 🔥 разница
            self.reconstruction_time = elapsed_anim  # сохрани как атрибут
            self.label.setText(f"Реконструкция завершена ✅ ({elapsed_anim:.2f} сек)")
            self.update_export_buttons()

    def open_photo_review(self):
        if not self.image_dir:
            self.label.setText("Сначала выберите папку с изображениями ❌")
            return

        low_quality_images = self.detect_low_quality_images(self.image_dir)
        print("[DEBUG] Плохие фото:", low_quality_images)

        used_images = [
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm'))
        ]

        from gui.photo_review_window import PhotoReviewWindow
        dialog = PhotoReviewWindow(
            image_dir=self.image_dir,
            used_images=used_images,
            low_quality_images=low_quality_images
        )
        dialog.exec_()

    def detect_low_quality_images(self, image_dir):
        import os
        import cv2

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        low_quality_paths = []
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm'))
        ])
        threshold = 200  # можно регулировать

        for i in range(len(image_files) - 1):
            img1_path = os.path.join(image_dir, image_files[i])
            img2_path = os.path.join(image_dir, image_files[i + 1])
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                continue

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                low_quality_paths.extend([img1_path, img2_path])
                continue

            matches = bf.match(des1, des2)
            if len(matches) < threshold:
                low_quality_paths.extend([img1_path, img2_path])

        return list(set(low_quality_paths))



