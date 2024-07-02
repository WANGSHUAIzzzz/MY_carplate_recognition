import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from plate_process import load_image, preprocess_image, detect_license_plate
from char_recognize import main as recognize_plate
from database_connection import check_license_plate

# 界面配置 Interface configuration
ctk.set_appearance_mode("dark")  # 设置深色模式 Set dark mode
ctk.set_default_color_theme("dark-blue")

class LicensePlateApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("License Plate Recognition System")
        self.geometry("1400x800")
        self.configure(bg="#1c1c1c")

        # 设置背景图片 Set background image
        self.bg_image = Image.open("background.png")  # 加载背景图片 Load background image
        self.bg_image = self.bg_image.resize((1400, 800), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.bg_image)
        
        self.bg_label = tk.Label(self, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        # 左侧图片显示区域 Left-side image display area
        self.left_frame = ctk.CTkFrame(self, width=800, height=600, fg_color="#2c2c2c", corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=5, padx=20, pady=20, sticky="n")

        self.original_image_label = ctk.CTkLabel(self.left_frame, text="Original Image", text_color="#ffffff", bg_color="#2c2c2c", font=("Helvetica", 16, "bold"))
        self.original_image_label.pack(padx=10, pady=10)

        self.original_image_canvas = tk.Canvas(self.left_frame, width=780, height=580, bg="#2c2c2c", highlightthickness=0)
        self.original_image_canvas.pack(padx=10, pady=10)

        # 右侧结果显示区域 Right-side result display area
        self.right_frame = ctk.CTkFrame(self, width=400, height=600, fg_color="#2c2c2c", corner_radius=0)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")

        self.binary_image_label = ctk.CTkLabel(self.right_frame, text="Binary Plate Image:", text_color="#ffffff", bg_color="#2c2c2c", font=("Helvetica", 16, "bold"))
        self.binary_image_label.pack(padx=10, pady=10)
        self.binary_image_canvas = tk.Canvas(self.right_frame, width=380, height=100, bg="#2c2c2c", highlightthickness=0)
        self.binary_image_canvas.pack(padx=10, pady=10)

        self.svm_result_label = ctk.CTkLabel(self.right_frame, text="SVM Detection Result (Can be wrong):", text_color="#ffffff", bg_color="#2c2c2c", font=("Helvetica", 16, "bold"))
        self.svm_result_label.pack(padx=10, pady=10)
        self.svm_result_value = ctk.CTkLabel(self.right_frame, text="", text_color="#00ff00", bg_color="#2c2c2c", font=("Helvetica", 14))
        self.svm_result_value.pack(padx=10, pady=10)

        self.ocr_result_label = ctk.CTkLabel(self.right_frame, text="OCR Recognition Result:", text_color="#ffffff", bg_color="#2c2c2c", font=("Helvetica", 16, "bold"))
        self.ocr_result_label.pack(padx=10, pady=10)
        self.ocr_result_value = ctk.CTkLabel(self.right_frame, text="", text_color="#00ff00", bg_color="#2c2c2c", font=("Helvetica", 14))
        self.ocr_result_value.pack(padx=10, pady=10)

        self.database_label = ctk.CTkLabel(self.right_frame, text="In Database:", text_color="#ffffff", bg_color="#2c2c2c", font=("Helvetica", 16, "bold"))
        self.database_label.pack(padx=10, pady=10)
        self.database_value = ctk.CTkLabel(self.right_frame, text="", text_color="#00ff00", bg_color="#2c2c2c", font=("Helvetica", 14))
        self.database_value.pack(padx=10, pady=10)

        self.allow_label = ctk.CTkLabel(self.right_frame, text="Allow", fg_color="#444444", width=200, height=50, corner_radius=0, font=("Helvetica", 14, "bold"))
        self.allow_label.pack(side="left", padx=10, pady=10)

        self.intercept_label = ctk.CTkLabel(self.right_frame, text="Intercept", fg_color="#444444", width=200, height=50, corner_radius=0, font=("Helvetica", 14, "bold"))
        self.intercept_label.pack(side="right", padx=10, pady=10)

        # 底部按钮 Bottom buttons
        self.bottom_frame = ctk.CTkFrame(self, width=1200, height=100, fg_color="#1c1c1c", corner_radius=0)
        self.bottom_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky="n")

        self.select_button = ctk.CTkButton(self.bottom_frame, text="Select Image", command=self.select_image, width=150, height=50, corner_radius=0, text_color="#ffffff", fg_color="#444444", hover_color="#555555", font=("Helvetica", 14, "bold"))
        self.select_button.pack(side="left", padx=10, pady=10)

        self.clear_button = ctk.CTkButton(self.bottom_frame, text="Clear Image", command=self.clear_image, width=150, height=50, corner_radius=0, text_color="#ffffff", fg_color="#444444", hover_color="#555555", font=("Helvetica", 14, "bold"))
        self.clear_button.pack(side="left", padx=10, pady=10)

        self.exit_button = ctk.CTkButton(self.bottom_frame, text="Exit", command=self.exit_app, width=150, height=50, corner_radius=0, text_color="#ffffff", fg_color="#444444", hover_color="#555555", font=("Helvetica", 14, "bold"))
        self.exit_button.pack(side="left", padx=10, pady=10)

    # 选择图片并进行处理 Select an image and process it
    def select_image(self):
        self.clear_image()  # 清除工作台图片 Clear workspace images first
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                raw_image = load_image(file_path)
                self.show_image(raw_image, self.original_image_canvas)

                filtered_img = preprocess_image(raw_image)
                plate_img = detect_license_plate(filtered_img, raw_image)

                if plate_img is not None:
                    binary_plate_img_path = './car_license/test.jpg'
                    if os.path.exists(binary_plate_img_path):
                        binary_plate_img = cv2.imread(binary_plate_img_path, cv2.IMREAD_GRAYSCALE)
                        self.show_image(binary_plate_img, self.binary_image_canvas, binary=True)

                    svm_result, ocr_result, error_message = recognize_plate()
                    if error_message:
                        messagebox.showwarning("Warning", error_message)
                        return

                    self.svm_result_value.configure(text=svm_result)
                    self.ocr_result_value.configure(text=ocr_result)
                    
                    is_recorded, allow_entry = check_license_plate(ocr_result)
                    self.database_value.configure(text="YES" if is_recorded else "NO")
                    if allow_entry:
                        self.allow_label.configure(fg_color="green")
                        self.intercept_label.configure(fg_color="gray40")
                    else:
                        self.allow_label.configure(fg_color="gray40")
                        self.intercept_label.configure(fg_color="red")
                else:
                    messagebox.showwarning("Warning", "No Plate Detected")

            except TypeError as e:
                messagebox.showwarning("Warning", "No Plate Detected")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # 清除图片 Clear the displayed images
    def clear_image(self):
        self.original_image_canvas.delete("all")
        self.binary_image_canvas.delete("all")
        self.svm_result_value.configure(text="")
        self.ocr_result_value.configure(text="")
        self.database_value.configure(text="")
        self.allow_label.configure(fg_color="gray40")
        self.intercept_label.configure(fg_color="gray40")

    # 显示图片 Display the image
    def show_image(self, img, canvas, binary=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if not binary:
            img.thumbnail((780, 580), Image.LANCZOS)
        else:
            img = img.resize((380, 100), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
        canvas.image = imgtk

    # 退出程序 Exit the application
    def exit_app(self):
        self.quit()

if __name__ == "__main__":
    app = LicensePlateApp()
    app.mainloop()
