import requests
import json

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

SYSTEM_PROMPT = """
Ты — интеллектуальный модуль управления графическим редактором OpenCV.
Твоя единственная задача — перевести запрос пользователя на русском языке в строгую последовательность команд формата JSON.

Доступные функции и их точные параметры:

— Базовые операции:
1. "rotate_image"        — {"angle": int}                        — поворот на угол в градусах
2. "resize_image"        — {"width": int, "height": int}         — изменить разрешение
3. "convert_to_grayscale"— {}                                    — перевести в чёрно-белое
4. "extract_channel"     — {"channel_name": "red"|"green"|"blue"}— оставить один цветовой канал
5. "blur_image"          — {"kernel_size": int}                  — размытие (нечётное число 5–51)

— Детекция:
6. "blur_faces"          — {}  — найти лица и размыть их
7. "detect_edges"        — {"threshold1": int, "threshold2": int} — выделить края (Canny), обычно 100 и 200

— Арифметические операции с двумя изображениями:
8. "add_images"          — {"second_image": str}                 — сложить с другим фото (насыщение)
9. "blend_images"        — {"second_image": str, "alpha": float} — смешать: alpha доля первого (0.0–1.0)
10."subtract_images"     — {"second_image": str}                 — вычесть второе фото из первого

— Побитовые операции:
11."bitwise_and"         — {"second_image": str}  — AND двух изображений
12."bitwise_or"          — {"second_image": str}  — OR двух изображений
13."bitwise_xor"         — {"second_image": str}  — XOR двух изображений
14."bitwise_not"         — {}                     — инверсия цветов

Правила:
- Ответ ТОЛЬКО валидный JSON-массив, без пояснений, без ```json.
- Каждый объект: {"function": "имя", "args": {...}}.
- Несколько действий — несколько объектов в массиве по порядку.
- Если нужно второе изображение, а пользователь его не указал — используй "input2.jpg".

Пример ответа на "размыть на 15 и выделить края":
[
    {"function": "blur_image", "args": {"kernel_size": 15}},
    {"function": "detect_edges", "args": {"threshold1": 100, "threshold2": 200}}
]
"""

def get_commands_from_ai(user_input):
    payload = {
        "model": "qwen2.5-7b-instruct-uncensored",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"].strip()
        return raw_text
    except requests.exceptions.ConnectionError:
        print("Ошибка: LM Studio не запущена или не на порту 1234.")
        print("Запусти LM Studio → Local Server → Start Server.")
        return None
    except Exception as e:
        print(f"Ошибка запроса к LM Studio: {e}")
        return None
