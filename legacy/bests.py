import os
import cv2
import numpy as np
from features import find_correspondence_points

image_dir = '../imgs/dinos'

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm', '.bmp', '.tif', '.tiff'))
])

matches_info = []

for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
        img_path1 = os.path.join(image_dir, image_files[i])
        img_path2 = os.path.join(image_dir, image_files[j])

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            continue

        try:
            pts1, pts2 = find_correspondence_points(img1, img2)
            num_matches = len(pts1.T)
        except Exception as e:
            print(f"Ошибка при обработке пары {image_files[i]} и {image_files[j]}: {e}")
            num_matches = 0

        print(f"{image_files[i]} + {image_files[j]} → {num_matches} совпадений")
        matches_info.append((image_files[i], image_files[j], num_matches))

# Сортируем по количеству совпадений
matches_info.sort(key=lambda x: -x[2])

print("\n📊 Топ лучших пар по количеству совпадений:")
for i, (img1, img2, m) in enumerate(matches_info[:10]):
    print(f"{i+1}) {img1} + {img2} → {m} совпадений")
