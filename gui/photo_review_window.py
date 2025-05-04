from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QScrollArea, QWidget, QGridLayout, QPushButton, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import os
import cv2


class PhotoReviewWindow(QDialog):
    def __init__(self, image_dir, used_images=None, low_quality_images=None):
        super().__init__()
        self.setWindowTitle("Обзор и коррекция фото")
        self.setMinimumSize(800, 600)

        self.image_dir = image_dir
        self.used_images = used_images or []
        self.low_quality_images = low_quality_images or []
        self.image_adjustments = {}
        self.selected_index = None

        self.scroll = QScrollArea()
        self.container = QWidget()
        self.grid = QGridLayout()
        self.container.setLayout(self.grid)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.container)

        self.btn_save = QPushButton("💾 Сохранить изменения")
        self.btn_save.clicked.connect(self.save_adjustments)

        self.btn_delete = QPushButton("🗑️ Удалить выбранное фото")
        self.btn_delete.clicked.connect(self.delete_selected_image)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_delete)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setMinimum(5)
        self.slider_brightness.setMaximum(20)
        self.slider_brightness.setValue(10)
        self.slider_brightness.valueChanged.connect(self.update_selected_image)
        layout.addWidget(QLabel("\u042f\u0440\u043a\u043e\u0441\u0442\u044c"))
        layout.addWidget(self.slider_brightness)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setMinimum(5)
        self.slider_contrast.setMaximum(20)
        self.slider_contrast.setValue(10)
        self.slider_contrast.valueChanged.connect(self.update_selected_image)
        layout.addWidget(QLabel("\u041a\u043e\u043d\u0442\u0440\u0430\u0441\u0442"))
        layout.addWidget(self.slider_contrast)

        self.setLayout(layout)
        self.load_images()

    def load_images(self):
        self.image_labels = []
        self.images = []

        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".pgm"))
        ])

        for i, filename in enumerate(image_files):
            path = os.path.join(self.image_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            self.images.append((img, path))
            self.image_adjustments[i] = {'brightness': 1.0, 'contrast': 1.0}

            label = ClickableLabel(index=i)
            label.setFixedSize(200, 150)
            label.setAlignment(Qt.AlignCenter)
            label.clicked.connect(self.image_selected)
            self.image_labels.append(label)

            if path in self.low_quality_images:
                label.setStyleSheet("border: 2px solid red;")
            elif path not in self.used_images:
                label.setStyleSheet("border: 2px solid gray;")
            else:
                label.setStyleSheet("border: none;")

            self.grid.addWidget(label, i // 4, i % 4)

        self.update_all_images()

    def image_selected(self, index):
        self.selected_index = index
        for i, label in enumerate(self.image_labels):
            if i == index:
                label.setStyleSheet("border: 3px solid blue;")
            else:
                path = self.images[i][1]
                if path in self.low_quality_images:
                    label.setStyleSheet("border: 2px solid red;")
                elif path not in self.used_images:
                    label.setStyleSheet("border: 2px solid gray;")
                else:
                    label.setStyleSheet("border: none;")

        settings = self.image_adjustments[index]
        self.slider_brightness.setValue(int(settings['brightness'] * 10))
        self.slider_contrast.setValue(int(settings['contrast'] * 10))
        self.update_selected_image()

    def update_selected_image(self):
        idx = self.selected_index
        if idx is None or idx >= len(self.images):
            return
        brightness = self.slider_brightness.value() / 10.0
        contrast = self.slider_contrast.value() / 10.0
        self.image_adjustments[idx] = {'brightness': brightness, 'contrast': contrast}
        self._update_image_at_index(idx)

    def update_all_images(self):
        for i in range(len(self.images)):
            self._update_image_at_index(i)

    def _update_image_at_index(self, i):
        img, _ = self.images[i]
        adj = self.image_adjustments[i]
        img_adj = cv2.convertScaleAbs(img, alpha=adj['contrast'], beta=50 * (adj['brightness'] - 1.0))
        rgb = cv2.cvtColor(img_adj, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(200, 150, Qt.KeepAspectRatio)
        self.image_labels[i].setPixmap(pix)

    def save_adjustments(self):
        for i, (img, path) in enumerate(self.images):
            try:
                adj = self.image_adjustments.get(i, {'brightness': 1.0, 'contrast': 1.0})
                img_adj = cv2.convertScaleAbs(img, alpha=adj['contrast'], beta=50 * (adj['brightness'] - 1.0))
                cv2.imwrite(path, img_adj)
            except Exception as e:
                print(f"[ERROR] Не удалось сохранить {path}: {e}")
        print("[INFO] Все изменения сохранены.")
        self.update_all_images()

    def delete_selected_image(self):
        if self.selected_index is None:
            return
        _, path = self.images[self.selected_index]
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[INFO] Удалено изображение: {path}")
            else:
                print(f"[WARNING] Файл не найден: {path}")
        except Exception as e:
            print(f"[ERROR] Не удалось удалить файл: {e}")

        self.selected_index = None  # 🛑 СБРОСИТЬ индекс ДО обновления
        self.load_images()


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, index=0, parent=None):
        super().__init__(parent)
        self.index = index

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)
