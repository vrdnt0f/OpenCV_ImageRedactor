import cv2
import numpy as np
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    def save(self, output_path="result.jpg"):
        cv2.imwrite(output_path, self.img)
        return f"Файл сохранён как '{output_path}'"

    # ── Вспомогательный метод: загрузить второе изображение ──────────────────
    def _load_second(self, second_image: str):
        img2 = cv2.imread(second_image)
        if img2 is None:
            raise FileNotFoundError(f"Второе изображение не найдено: {second_image}")
        if img2.shape != self.img.shape:
            img2 = cv2.resize(img2, (self.img.shape[1], self.img.shape[0]))
        return img2

    # ── 1. Поворот ────────────────────────────────────────────────────────────
    def rotate_image(self, angle: int):
        mapping = {90: cv2.ROTATE_90_CLOCKWISE,
                   180: cv2.ROTATE_180,
                   270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                   -90: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if angle in mapping:
            self.img = cv2.rotate(self.img, mapping[angle])
        else:
            h, w = self.img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
            self.img = cv2.warpAffine(self.img, M, (w, h))
        return f"Поворот на {angle}° выполнен."

    # ── 2. Изменение размера ──────────────────────────────────────────────────
    def resize_image(self, width: int, height: int):
        self.img = cv2.resize(self.img, (width, height), interpolation=cv2.INTER_AREA)
        return f"Размер изменён на {width}×{height}."

    # ── 3. Чёрно-белое ───────────────────────────────────────────────────────
    def convert_to_grayscale(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return "Переведено в оттенки серого."

    # ── 4. Выделение канала ───────────────────────────────────────────────────
    def extract_channel(self, channel_name: str):
        b, g, r = cv2.split(self.img)
        z = np.zeros_like(b)
        ch = channel_name.lower()
        if ch in ("red", "красный"):
            self.img = cv2.merge([z, z, r])
        elif ch in ("green", "зелёный", "зеленый"):
            self.img = cv2.merge([z, g, z])
        elif ch in ("blue", "синий"):
            self.img = cv2.merge([b, z, z])
        else:
            return f"Канал '{channel_name}' не поддерживается (red/green/blue)."
        return f"Оставлен только канал '{channel_name}'."

    # ── 5. Размытие ───────────────────────────────────────────────────────────
    def blur_image(self, kernel_size: int):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)
        return f"Гауссово размытие с ядром {kernel_size} применено."

    # ── 6. Размытие лиц ───────────────────────────────────────────────────────
    def blur_faces(self, cascade_path="face.xml"):
        if not os.path.exists(cascade_path):
            return f"Ошибка: файл каскада '{cascade_path}' не найден."
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = self.img[y:y+h, x:x+w]
            self.img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)
        return f"Найдено и размыто лиц: {len(faces)}."

    # ── 7. Выделение краёв (Canny) ────────────────────────────────────────────
    def detect_edges(self, threshold1: int = 100, threshold2: int = 200):
        edges = cv2.Canny(self.img, threshold1, threshold2)
        self.img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return f"Края выделены (Canny {threshold1}/{threshold2})."

    # ── 8. Сложение двух изображений ─────────────────────────────────────────
    def add_images(self, second_image: str):
        img2 = self._load_second(second_image)
        self.img = cv2.add(self.img, img2)
        return f"Изображения сложены (+ {second_image})."

    # ── 9. Смешивание с весами ────────────────────────────────────────────────
    def blend_images(self, second_image: str, alpha: float = 0.7):
        img2 = self._load_second(second_image)
        self.img = cv2.addWeighted(self.img, alpha, img2, 1.0 - alpha, 0)
        return f"Смешано: {alpha:.0%} первого + {1-alpha:.0%} второго ({second_image})."

    # ── 10. Вычитание ────────────────────────────────────────────────────────
    def subtract_images(self, second_image: str):
        img2 = self._load_second(second_image)
        self.img = cv2.subtract(self.img, img2)
        return f"Вычтено второе изображение ({second_image})."

    # ── 11. Побитовое AND ────────────────────────────────────────────────────
    def bitwise_and(self, second_image: str):
        img2 = self._load_second(second_image)
        self.img = cv2.bitwise_and(self.img, img2)
        return f"Побитовое AND с {second_image}."

    # ── 12. Побитовое OR ─────────────────────────────────────────────────────
    def bitwise_or(self, second_image: str):
        img2 = self._load_second(second_image)
        self.img = cv2.bitwise_or(self.img, img2)
        return f"Побитовое OR с {second_image}."

    # ── 13. Побитовое XOR ────────────────────────────────────────────────────
    def bitwise_xor(self, second_image: str):
        img2 = self._load_second(second_image)
        self.img = cv2.bitwise_xor(self.img, img2)
        return f"Побитовое XOR с {second_image}."

    # ── 14. Инверсия (NOT) ───────────────────────────────────────────────────
    def bitwise_not(self):
        self.img = cv2.bitwise_not(self.img)
        return "Цвета инвертированы (NOT)."
