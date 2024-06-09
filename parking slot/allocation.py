import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk
import cv2
from ttkthemes import ThemedStyle  # Import ThemedStyle from ttkthemes
from ultralytics import YOLO


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DETECTION")
        self.root.geometry("800x800")  # Set window size

        style = ThemedStyle(root)
        style.set_theme("equilux")  # Set the theme for the whole app

        self.tab_control = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text='PARKING SLOT DETECTION')
        self.tab_control.add(self.tab2, text='Allocate slot')
        self.tab_control.pack(expand=1, fill='both')

        title_label = tk.Label(self.tab1, text="PARKING SLOT DETECTION ", font=("Georgia", 24, "bold italic"))
        title_label.pack(pady=20)

        self.load_button = tk.Button(self.tab1, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(self.tab1, text="Detect", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.image_label_tab1 = tk.Label(self.tab1, bg="LightYellow", padx=20, pady=20)  # Set panel color and padding
        self.image_label_tab1.pack(fill=tk.BOTH, expand=True)  # Expand label to fill window

        self.loaded_image = None
        self.loaded_cv_image = None

        # Load YOLO model
        self.model = YOLO("runs/detect/train2/weights/best.pt")  # Placeholder for YOLO model, will be initialized later

        # Display a constant image in the "Allocate slot" tab
        self.image_label_tab2 = tk.Label(self.tab2, bg="LightYellow", padx=20, pady=20)
        self.image_label_tab2.pack(fill=tk.BOTH, expand=True)
        self.display_constant_image()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.loaded_image = Image.open(file_path)
            self.loaded_image = self.loaded_image.resize((400, 400))  # Resize image if needed
            self.loaded_cv_image = cv2.cvtColor(np.array(self.loaded_image), cv2.COLOR_RGB2BGR)
            self.display_cv_image(self.loaded_cv_image, self.image_label_tab1)

    def detect_objects(self):
        if self.loaded_cv_image is not None:
            if self.model is not None:
                try:
                    results = self.model(source=self.loaded_cv_image)
                    res_plotted = results[0].plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    self.display_cv_image(res_plotted_rgb, self.image_label_tab1)
                except Exception as e:
                    self.display_message(f"Error detecting objects: {e}")
            else:
                self.display_message("Model not loaded!")

    def display_cv_image(self, cv_image, label_widget):
        cv_image = Image.fromarray(cv_image)
        cv_image = cv_image.resize((600, 600))
        photo = ImageTk.PhotoImage(cv_image)
        label_widget.config(image=photo)
        label_widget.image = photo

    def display_message(self, message):
        self.image_label_tab1.config(text=message)

    def check_availability(self):
        # Add code here to check availability and display it in the Availability tab
        pass

    def display_constant_image(self):
        # Load and display the constant image in the "Allocate slot" tab
        constant_image_path = "train/images/31_jpg.rf.a11155de60586e610c636255291ae09d.jpg"
        constant_image = Image.open(constant_image_path)
        constant_image = constant_image.resize((400, 400), Image.ANTIALIAS)  # Resize image if needed
        constant_image = ImageTk.PhotoImage(constant_image)
        self.image_label_tab2.configure(image=constant_image)
        self.image_label_tab2.image = constant_image

        # Add text indicating available slots
        available_slots_text = "Available slots are 11, 26, and 22"
        slots_label = tk.Label(self.tab2, text=available_slots_text, font=("Arial", 12, "bold"))
        slots_label.place(relx=0.5, rely=0.05, anchor="center")

        # Add text indicating the allocated slot
        allocated_slot_text = "Allocated slot: 22"
        allocated_slot_label = tk.Label(self.tab2, text=allocated_slot_text, font=("Arial", 12, "bold"),fg="green")
        allocated_slot_label.place(relx=0.5, rely=0.1, anchor="center")

        # Add text indicating remaining available slots
        remaining_slots_text = "Remaining slots: 11 and 26"
        remaining_slots_label = tk.Label(self.tab2, text=remaining_slots_text, font=("Arial", 12, "bold"), fg="blue")
        remaining_slots_label.place(relx=0.5, rely=0.15, anchor="center")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()