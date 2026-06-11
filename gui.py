import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import json
import os
from PIL import Image, ImageTk
import cv2
import numpy as np

from processor import ImageProcessor
from ai_bridge import get_commands_from_ai

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

BG        = "#1a1a2e"
PANEL     = "#16213e"
CARD      = "#0f3460"
ACCENT    = "#e94560"
ACCENT2   = "#53d8fb"
TEXT      = "#eaeaea"
TEXT_DIM  = "#8892a4"
SUCCESS   = "#4ecca3"
ERROR     = "#e94560"
FONT_UI   = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)


def _imread_pil(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def fit_image(pil_img, max_w, max_h):
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return pil_img.resize((nw, nh), Image.LANCZOS)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenCV AI Редактор")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 600)

        self.input_path   = None
        self.second_path  = None
        self.output_path  = None
        self._tk_src      = None
        self._tk_dst      = None

        self._build_ui()
        self.update_idletasks()
        self.geometry("1100x700")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──
        top = tk.Frame(self, bg=PANEL, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="⬡ OpenCV AI Редактор", font=("Segoe UI", 14, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=18)
        tk.Label(top, text="управление изображением на русском языке",
                 font=("Segoe UI", 9), bg=PANEL, fg=TEXT_DIM).pack(side="left")

        # ── main area ──
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=(10, 6))
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.frame_src = self._image_card(main, "Исходное изображение", 0)
        self.frame_dst = self._image_card(main, "Результат",            1)

        # ── bottom panel ──
        bottom = tk.Frame(self, bg=PANEL, pady=10)
        bottom.pack(fill="x", padx=16, pady=(0, 10))

        # file button
        self.btn_file = tk.Button(
            bottom, text="📂  Открыть", font=FONT_UI,
            bg=CARD, fg=ACCENT2, activebackground=ACCENT, activeforeground=TEXT,
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._open_file)
        self.btn_file.pack(side="left", padx=(0, 6))

        # second image button
        self.btn_file2 = tk.Button(
            bottom, text="📂  Второе фото", font=FONT_UI,
            bg=CARD, fg=TEXT_DIM, activebackground=ACCENT, activeforeground=TEXT,
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._open_second)
        self.btn_file2.pack(side="left", padx=(0, 10))

        # command entry
        self.entry = tk.Entry(
            bottom, font=("Segoe UI", 11),
            bg="#0d2137", fg=TEXT, insertbackground=ACCENT2,
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground=CARD, highlightcolor=ACCENT2)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 10))
        self.entry.insert(0, "Введите команду, например: поверни на 90 и сделай чб")
        self.entry.bind("<FocusIn>",  self._clear_hint)
        self.entry.bind("<Return>",   lambda e: self._run())

        # run button
        self.btn_run = tk.Button(
            bottom, text="▶  Выполнить", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg=TEXT, activebackground="#c73652", activeforeground=TEXT,
            relief="flat", padx=16, pady=6, cursor="hand2",
            command=self._run)
        self.btn_run.pack(side="left")

        # ── log strip ──
        self.log_var = tk.StringVar(value="Выберите изображение и введите команду.")
        tk.Label(self, textvariable=self.log_var, font=FONT_MONO,
                 bg=BG, fg=TEXT_DIM, anchor="w").pack(fill="x", padx=18, pady=(0, 4))

    def _image_card(self, parent, title, col):
        frame = tk.Frame(parent, bg=CARD, bd=0)
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 8) if col == 0 else (8, 0), pady=4)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=TEXT_DIM, pady=6).grid(row=0, column=0)

        canvas = tk.Label(frame, bg=PANEL, text="—", fg=TEXT_DIM,
                          font=("Segoe UI", 28))
        canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        return canvas

    # ── helpers ───────────────────────────────────────────────────────────────
    def _clear_hint(self, event):
        if self.entry.get().startswith("Введите"):
            self.entry.delete(0, "end")

    def _log(self, msg, color=TEXT_DIM):
        self.log_var.set(msg)
        self.after(0, lambda: self.log_var.set(msg))

    def _show_image(self, label_widget, path, ref_attr):
        pil = _imread_pil(path)
        if pil is None:
            return
        label_widget.update_idletasks()
        w = max(label_widget.winfo_width(),  300)
        h = max(label_widget.winfo_height(), 300)
        pil_fit = fit_image(pil, w - 16, h - 16)
        tk_img = ImageTk.PhotoImage(pil_fit)
        setattr(self, ref_attr, tk_img)
        label_widget.config(image=tk_img, text="")

    # ── actions ───────────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        self.input_path = path
        self._show_image(self.frame_src, path, "_tk_src")
        self.frame_dst.config(image="", text="—")
        self._log(f"✓ Открыто: {os.path.basename(path)}", SUCCESS)

    def _open_second(self):
        path = filedialog.askopenfilename(
            title="Выберите второе изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        self.second_path = path
        name = os.path.basename(path)
        self.btn_file2.config(text=f"📂  {name}", fg=ACCENT2)
        self._log(f"✓ Второе фото: {name}")

    def _run(self):
        if not self.input_path:
            messagebox.showwarning("Нет файла", "Сначала откройте изображение.")
            return
        cmd = self.entry.get().strip()
        if not cmd or cmd.startswith("Введите"):
            messagebox.showwarning("Нет команды", "Введите команду на русском.")
            return
        self.btn_run.config(state="disabled", text="⏳  Обработка…")
        self._log("⏳ Отправляю запрос к нейросети…")
        threading.Thread(target=self._process, args=(cmd,), daemon=True).start()

    def _process(self, user_cmd):
        try:
            ai_resp = get_commands_from_ai(user_cmd)
            if not ai_resp:
                self.after(0, lambda: self._done_error("Нет ответа от LM Studio."))
                return

            self.after(0, lambda: self._log(f"🤖 JSON: {ai_resp[:120]}…" if len(ai_resp) > 120 else f"🤖 JSON: {ai_resp}"))

            try:
                commands = json.loads(ai_resp)
            except json.JSONDecodeError:
                self.after(0, lambda: self._done_error(f"Некорректный JSON от ИИ: {ai_resp[:80]}"))
                return

            processor = ImageProcessor(self.input_path)

            TWO_IMAGE_FUNCS = {
                "add_images", "blend_images", "subtract_images",
                "bitwise_and", "bitwise_or", "bitwise_xor"
            }

            results = []
            for cmd in commands:
                func_name = cmd.get("function", "")
                args = cmd.get("args", {})
                if func_name == "blur_faces":
                    args.setdefault("cascade_path", "face.xml")
                # подставляем второе изображение если выбрано
                if func_name in TWO_IMAGE_FUNCS and self.second_path:
                    args["second_image"] = self.second_path
                if hasattr(processor, func_name):
                    try:
                        msg = getattr(processor, func_name)(**args)
                        results.append(f"✓ {msg}")
                    except Exception as e:
                        results.append(f"✗ {func_name}: {e}")
                else:
                    results.append(f"✗ Неизвестная команда: {func_name}")

            base, ext = os.path.splitext(self.input_path)
            out = f"{base}_result{ext}"
            processor.save(out)
            self.output_path = out

            summary = " | ".join(results)
            self.after(0, lambda: self._done_ok(out, summary))

        except Exception as e:
            self.after(0, lambda: self._done_error(str(e)))

    def _done_ok(self, out_path, summary):
        self._show_image(self.frame_dst, out_path, "_tk_dst")
        self._log(f"✓ {summary}", SUCCESS)
        self.btn_run.config(state="normal", text="▶  Выполнить")

    def _done_error(self, msg):
        self._log(f"✗ {msg}", ERROR)
        self.btn_run.config(state="normal", text="▶  Выполнить")
        messagebox.showerror("Ошибка", msg)


if __name__ == "__main__":
    app = App()
    app.mainloop()
