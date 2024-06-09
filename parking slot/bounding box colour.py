import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from ttkthemes import ThemedStyle  # Import ThemedStyle from ttkthemes

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DETECTION")
        self.root.geometry("800x800")  # Set window size

        style = ThemedStyle(root)
        style.set_theme("equilux")  # Set the theme for the whole app

        title_label = tk.Label(root, text="DETECTION ", font=("Helvetica", 20))
        title_label.pack(pady=20)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="LightYellow", padx=20, pady=20)  # Set panel color and padding
        self.image_label.pack(fill=tk.BOTH, expand=True)  # Expand label to fill window

        self.loaded_image = None
        self.loaded_cv_image = None

        # Load YOLO model
        self.model = YOLO("runs/detect/train2/weights/best.pt")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.loaded_image = Image.open(file_path)
            self.loaded_image = self.loaded_image.resize((400, 400))  # Resize image if needed
            self.loaded_cv_image = cv2.cvtColor(np.array(self.loaded_image), cv2.COLOR_RGB2BGR)
            self.display_cv_image(self.loaded_cv_image)

    def detect_objects(self):
        if self.loaded_cv_image is not None:
            if self.model is not None:
                results = self.model(source=self.loaded_cv_image)
                # Check if results is a list
                if isinstance(results, list):
                    # Assume results contains bounding box information directly
                    for box in results:
                        print("Box:", box)  # Print out the box to understand its structure
                        # Modify this part to draw bounding boxes based on the structure of 'box'
                else:
                    # Assume results is a standard YOLO detection output
                    for box in results.xyxy[0]:
                        xmin, ymin, xmax, ymax, conf, cls = box
                        cv2.rectangle(self.loaded_cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0),
                                      2)
                self.display_cv_image(self.loaded_cv_image)
            else:
                self.display_message("Model not loaded!")

    def display_cv_image(self, cv_image):
        cv_image = Image.fromarray(cv_image)
        cv_image = cv_image.resize((600, 600))
        photo = ImageTk.PhotoImage(cv_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def display_message(self, message):
        self.image_label.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()