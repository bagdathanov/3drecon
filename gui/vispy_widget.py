# gui/vispy_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from vispy import scene
from vispy.scene import visuals
import numpy as np

class Vispy3DWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(800, 400))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up='z', elevation=135, azimuth=90, distance=2)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

    def set_point_cloud(self, points):
        if points.shape[0] == 4:
            points = points[:3, :] / points[3, :]
        elif points.shape[0] != 3:
            raise ValueError("Ожидались 3D или 4D точки")

        points = points.T
        points -= points.mean(axis=0)
        points /= np.max(np.abs(points))  # нормализация

        self.scatter.set_data(points, face_color='black', size=2.0)