"""
GUI for Medical Image Severity Classification
Allows users to upload a brain MRI image and get a severity prediction.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from hybrid_model import HybridVGG16Fuzzy
from utils import check_model_exists


class SeverityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical MRI Severity Classification")
        self.state('zoomed')
        self.configure(bg="#f2f6fb")

        self.model = None
        self.image_path = None
        self.image_preview = None

        self.create_widgets()

    def create_widgets(self):
        self.option_add("*Font", "Arial 11")

        header_frame = tk.Frame(self, bg="#1f4f81", padx=20, pady=18)
        header_frame.pack(fill=tk.X)

        header_label = tk.Label(
            header_frame,
            text="Medical MRI Severity Classification",
            font=("Arial", 26, "bold"),
            fg="#ffffff",
            bg="#1f4f81",
        )
        header_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Upload a single MRI image to predict severity and tumor risk.",
            font=("Arial", 14),
            fg="#d8e3f0",
            bg="#1f4f81",
            pady=6,
        )
        subtitle_label.pack()

        action_frame = tk.Frame(self, bg="#f2f6fb", pady=16)
        action_frame.pack(fill=tk.X)

        self.select_button = tk.Button(
            action_frame,
            text="Select MRI Image",
            font=("Arial", 13, "bold"),
            bg="#4b7bec",
            fg="#ffffff",
            activebackground="#335c8f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            command=self.select_image,
        )
        self.select_button.grid(row=0, column=0, padx=16)

        self.predict_button = tk.Button(
            action_frame,
            text="Run Prediction",
            font=("Arial", 13, "bold"),
            bg="#3b8fd6",
            fg="#ffffff",
            activebackground="#2b6ca5",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            state=tk.DISABLED,
            command=self.predict_image,
        )
        self.predict_button.grid(row=0, column=1, padx=16)

        self.status_label = tk.Label(
            action_frame,
            text="Please select an MRI image to begin.",
            font=("Arial", 12),
            fg="#333333",
            bg="#f2f6fb",
        )
        self.status_label.grid(row=0, column=2, padx=16, sticky="w")

        main_frame = tk.Frame(self, bg="#f2f6fb")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        preview_frame = tk.LabelFrame(main_frame, text="Image Preview", font=("Arial", 14, "bold"), fg="#1f4f81", bg="#ffffff", bd=2, relief=tk.GROOVE)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_label = tk.Label(preview_frame, text="No image selected.", font=("Arial", 12), pady=20, bg="#ffffff")
        self.preview_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        result_frame = tk.LabelFrame(main_frame, text="Prediction Result", font=("Arial", 14, "bold"), fg="#1f4f81", bg="#ffffff", bd=2, relief=tk.GROOVE)
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = tk.Text(
            result_frame,
            width=80,
            height=18,
            font=("Consolas", 12),
            wrap=tk.WORD,
            state=tk.DISABLED,
            bd=0,
            padx=12,
            pady=12,
            bg="#f9fbff",
            fg="#1b2838",
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        footer = tk.Label(
            self,
            text="The GUI runs predictions using the trained hybrid model in saved_model/.",
            font=("Arial", 10),
            fg="#555555",
            bg="#f2f6fb",
            pady=10,
        )
        footer.pack(fill=tk.X)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*")
            ]
        )

        if not file_path:
            return

        self.image_path = file_path
        self.update_preview(file_path)

        valid, score = self.check_mri_likelihood(file_path)
        if not valid:
            messagebox.showerror(
                "Not an MRI Image",
                "This image does not appear to be a brain MRI slice.\nPlease select a valid MRI image."
            )
            self.image_path = None
            self.preview_label.config(image="", text="No image selected.")
            self.predict_button.config(state=tk.DISABLED)
            self.status_label.config(text="Please select a valid MRI image.")
            return

        self.predict_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Image selected. MRI likelihood score: {score:.2f}. Click 'Run Prediction' to continue.")

    def update_preview(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            resample = getattr(Image, 'Resampling', Image)
            image.thumbnail((300, 300), resample.LANCZOS)
            self.image_preview = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self.image_preview, text="")
            self.preview_label.image = self.image_preview
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not load image preview:\n{e}")
            self.preview_label.config(text="Preview failed.")

    def check_mri_likelihood(self, image_path):
        """Run a basic heuristic to verify the image looks like a brain MRI slice."""
        image = cv2.imread(image_path)
        if image is None:
            return False, 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        value_mean = float(np.mean(hsv[:, :, 2]))

        b, g, r = cv2.split(image)
        color_diff = float(np.mean(np.abs(b - g) + np.abs(b - r) + np.abs(g - r))) / 3.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.sum(edges > 0)) / gray.size

        # MRI slices are typically low-saturation grayscale images with moderate edge structure.
        gray_score = max(0.0, 1.0 - (color_diff / 80.0))
        sat_score = max(0.0, 1.0 - (saturation_mean / 75.0))
        edge_score = min(edge_ratio * 10.0, 1.0)

        score = 0.5 * gray_score + 0.35 * sat_score + 0.15 * edge_score
        score = float(np.clip(score, 0.0, 1.0))

        valid = score >= 0.55
        return valid, score

    def load_model(self):
        if not check_model_exists():
            messagebox.showerror("Model Not Found", "Trained model files were not found in saved_model/. Please run training first.")
            return False

        try:
            self.status_label.config(text="Loading trained model... please wait.")
            self.update_idletasks()
            self.model = HybridVGG16Fuzzy()
            self.model.load('saved_model/')
            return True
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Unable to load trained model:\n{e}")
            return False

    def predict_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an MRI image first.")
            return

        if self.model is None:
            if not self.load_model():
                return

        self.status_label.config(text="Running prediction... please wait.")
        self.update_idletasks()

        result = self.model.predict_single(self.image_path)

        if result is None:
            messagebox.showerror("Prediction Failed", "The image could not be processed.")
            self.status_label.config(text="Prediction failed.")
            return

        self.display_result(result)
        self.status_label.config(text="Prediction complete.")

    def display_result(self, result):
        # Determine tumor likelihood from predicted severity
        severity_label = result['final_label']
        if severity_label in ['Normal', 'Mild']:
            tumor_risk = 'LOW'
            interpretation = 'Low likelihood of severe tumor progression.'
        elif severity_label == 'Moderate':
            tumor_risk = 'MEDIUM'
            interpretation = 'Moderate likelihood of tumor severity; follow-up recommended.'
        else:
            tumor_risk = 'HIGH'
            interpretation = 'High likelihood of severe tumor pathology; urgent review recommended.'

        text = []
        text.append(f"Image: {os.path.basename(self.image_path)}")
        text.append(f"Predicted Severity: {severity_label}")
        text.append(f"Fuzzy Score: {result['fuzzy_score']:.2f} / 3.0")
        text.append(f"Decision Method: {result.get('decision_method', 'Hybrid')}")
        text.append(f"Tumor Risk: {tumor_risk}")
        text.append(f"Interpretation: {interpretation}")
        text.append("\nCNN Confidence:")

        for label, prob in result['cnn_probabilities'].items():
            text.append(f"  {label:<9}: {prob*100:5.1f}%")

        text.append("\nFuzzy Logic Inputs:")
        fuzzy_inputs = result['fuzzy_inputs']
        text.append(f"  Tumor Intensity      : {fuzzy_inputs['tumor_intensity']:.3f}")
        text.append(f"  Texture Irregularity : {fuzzy_inputs['texture_irregularity']:.3f}")
        text.append(f"  Boundary Clarity     : {fuzzy_inputs['boundary_clarity']:.3f}")

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(text))
        self.result_text.config(state=tk.DISABLED)


if __name__ == '__main__':
    app = SeverityGUI()
    app.mainloop()
