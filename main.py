import json
import os
from processor import ImageProcessor
from ai_bridge import get_commands_from_ai

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def choose_image(prompt="Выберите изображение"):
    files = sorted([f for f in os.listdir('.') if f.lower().endswith(SUPPORTED_EXT)])
    if not files:
        print(f"  Ошибка: в папке нет изображений ({', '.join(SUPPORTED_EXT)})")
        return None
    print(f"\n{prompt}:")
    for i, name in enumerate(files, 1):
        print(f"  [{i}] {name}")
    while True:
        try:
            idx = int(input(f"Введите номер (1–{len(files)}): ").strip()) - 1
            if 0 <= idx < len(files):
                return files[idx]
            print(f"  Нужно число от 1 до {len(files)}.")
        except ValueError:
            print("  Неверный ввод, попробуйте снова.")

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  Система интеллектуального управления OpenCV ║")
    print("╚══════════════════════════════════════════════╝")

    input_file = choose_image("Выберите исходное изображение")
    if not input_file:
        return

    print(f"\n✓ Выбрано: {input_file}")

    user_query = input(
        "\nВведите команду на русском языке\n"
        "Примеры: 'поверни на 90 и сделай чб'\n"
        "         'смешай с input2.jpg, потом выдели края'\n"
        "         'размой лица'\n> "
    )

    print("\n[ИИ] Формирую план обработки...")
    ai_response = get_commands_from_ai(user_query)

    if not ai_response:
        print("[Ошибка] Нет ответа от нейросети.")
        return

    print(f"[ИИ] JSON-план:\n{ai_response}\n")

    try:
        commands = json.loads(ai_response)
    except json.JSONDecodeError:
        print("[Ошибка] Нейросеть выдала некорректный JSON.")
        print(f"Ответ был: {ai_response}")
        return

    try:
        processor = ImageProcessor(input_file)
    except FileNotFoundError as e:
        print(f"[Ошибка] {e}")
        return

    print("[OpenCV] Выполняю команды:")
    for cmd in commands:
        func_name = cmd.get("function", "")
        args = cmd.get("args", {})

        # Каскады подставляем автоматически
        if func_name == "blur_faces":
            args.setdefault("cascade_path", "face.xml")

        if hasattr(processor, func_name):
            try:
                msg = getattr(processor, func_name)(**args)
                print(f"  ✓ {msg}")
            except Exception as e:
                print(f"  ✗ Ошибка в '{func_name}': {e}")
        else:
            print(f"  ✗ Неизвестная команда: '{func_name}'")

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_result{ext}"
    print(f"\n[Готово] {processor.save(output_file)}")

if __name__ == "__main__":
    main()
