import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Button, Label, Entry
import os
import rasterio

def get_pixel_size(path):
    with rasterio.open(path) as src:
        pixel_width = src.transform.a
        pixel_height = abs(src.transform.e)
        return pixel_width, pixel_height

def scale_mask_to_target(mask_path, target_path):
    mask_pixel_size = get_pixel_size(mask_path)
    target_pixel_size = get_pixel_size(target_path)

    print(f"Mask pixel scale before scaling: {mask_pixel_size}")
    print(f"Target pixel scale: {target_pixel_size}")

    scale_x = mask_pixel_size[0] / target_pixel_size[0]
    scale_y = mask_pixel_size[1] / target_pixel_size[1]

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    new_size = (
        int(mask_img.shape[1] * scale_x),
        int(mask_img.shape[0] * scale_y)
    )
    scaled_mask = cv2.resize(mask_img, new_size, interpolation=cv2.INTER_NEAREST)

    # Print pixel scale after conversion (now matches target)
    print(f"Mask pixel scale after scaling: {target_pixel_size}")
    return scaled_mask

class MaskFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Find Mask On Parcels Image")
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.mask_img = None
        self.target_img = None
        self.tk_image = None
        self.preprocessed_img = None

        # Zoom and pan state
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_mouse_pos = None
        self.display_img = None  # The image currently displayed (target, mask, or processed)

        # Default paths
        root_dir = os.path.dirname(os.path.abspath(__file__))
        default_mask_path = os.path.join(root_dir, "ExportedScaledMask.tif")
        default_target_path = os.path.join(root_dir, "Southport[96dpi].tif")

        # Entry fields for image paths
        self.mask_path_entry = Entry(root, width=120)
        self.mask_path_entry.insert(0, default_mask_path)
        self.mask_path_entry.pack(side=tk.TOP, padx=5, pady=2)
        self.target_path_entry = Entry(root, width=120)
        self.target_path_entry.insert(0, default_target_path)
        self.target_path_entry.pack(side=tk.TOP, padx=5, pady=2)

        # Buttons
        self.load_target_btn = Button(root, text="1-Load Target Image", command=self.load_target_image)
        self.load_target_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.load_mask_btn = Button(root, text="2-Load Mask Image", command=self.load_mask_image)
        self.load_mask_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # self.preprocess_btn = Button(root, text="Preprocess Target (Contours & Thicken)", command=self.preprocess_target_image)
        # self.preprocess_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.find_btn = Button(root, text="3-Find Mask Candidates", command=self.find_and_highlight_mask_candidates)
        self.find_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_extents_btn = Button(root, text="Zoom to Extents", command=self.zoom_to_extents)
        self.zoom_extents_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = Label(root, text="Load images to begin.")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Threshold entry
        self.threshold_entry = Entry(root)
        self.threshold_entry.pack(side=tk.TOP, padx=5, pady=2)
        self.threshold_entry.insert(0, "0.8")  # Default threshold

        # Method selection
        # self.method_var = tk.StringVar(value="TM_CCOEFF_NORMED")
        # methods = ["TM_CCOEFF_NORMED", "TM_SQDIFF_NORMED", "TM_CCORR_NORMED"]
        # for method in methods:
        #     radiobutton = tk.Radiobutton(root, text=method, variable=self.method_var, value=method)
        #     radiobutton.pack(side=tk.TOP, padx=5, pady=2)

        # Bind zoom and pan events
        self.canvas.bind("<MouseWheel>", self.zoom_event)  # Windows
        self.canvas.bind("<Button-4>", self.zoom_event)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom_event)    # Linux scroll down
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_event)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        # Optional: Shift+Left for pan (for touchpads)
        self.canvas.bind("<ButtonPress-1>", self.start_pan_shift)
        self.canvas.bind("<B1-Motion>", self.pan_event_shift)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan_shift)

    def show_image(self, img):
        """Display the current image with zoom and pan."""
        if img is None:
            return
        self.display_img = img
        h, w = img.shape[:2]
        # Calculate view window
        view_w = int(self.canvas_width / self.zoom)
        view_h = int(self.canvas_height / self.zoom)
        x1 = int(self.offset_x)
        y1 = int(self.offset_y)
        x2 = min(x1 + view_w, w)
        y2 = min(y1 + view_h, h)
        # Crop and resize
        cropped = img[y1:y2, x1:x2]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return
        resized = cv2.resize(cropped, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_AREA)
        # Convert for Tkinter
        if len(resized.shape) == 2:
            preview_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            preview_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def load_mask_image(self):
        file_path = self.mask_path_entry.get()
        if not os.path.isfile(file_path):
            file_path = filedialog.askopenfilename(initialdir=os.path.dirname(file_path), filetypes=[("Image Files", "*.tif;*.png;*.jpg;*.bmp")])
            if not file_path:
                self.status_label.config(text="Mask image not loaded.")
                return
            self.mask_path_entry.delete(0, tk.END)
            self.mask_path_entry.insert(0, file_path)
        # Print pixel scale on load
        try:
            mask_pixel_size = get_pixel_size(file_path)
            print(f"Mask pixel scale on load: {mask_pixel_size}")
        except Exception as e:
            print(f"Could not read mask pixel scale: {e}")
        # Wait until target image is loaded
        if self.target_img is not None:
            target_path = self.target_path_entry.get()
            self.mask_img = scale_mask_to_target(file_path, target_path)
            self.overlay_mask_bottom_right()  # <-- Overlay mask after scaling
        else:
            self.mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.mask_img is not None:
                self.status_label.config(text=f"Loaded mask image: {file_path}")
                self.display_img = self.mask_img
                self.zoom_to_extents()
            else:
                self.status_label.config(text="Failed to load mask image.")

    def load_target_image(self):
        file_path = self.target_path_entry.get()
        if not os.path.isfile(file_path):
            file_path = filedialog.askopenfilename(initialdir=os.path.dirname(file_path), filetypes=[("Image Files", "*.tif;*.png;*.jpg;*.bmp")])
            if not file_path:
                self.status_label.config(text="Target image not loaded.")
                return
            self.target_path_entry.delete(0, tk.END)
            self.target_path_entry.insert(0, file_path)
        # Print pixel scale on load
        try:
            target_pixel_size = get_pixel_size(file_path)
            print(f"Target pixel scale on load: {target_pixel_size}")
        except Exception as e:
            print(f"Could not read target pixel scale: {e}")
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.target_img = img
            self.status_label.config(text=f"Loaded target image: {file_path}")
            self.display_img = self.target_img
            self.zoom_to_extents()
        else:
            self.status_label.config(text="Failed to load target image.")

    def preprocess_target_image(self):
        if self.target_img is None:
            self.status_label.config(text="Target image not loaded.")
            return
        ret, thresh = cv2.threshold(self.target_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thick = 5
        contour_img = np.zeros_like(self.target_img)
        cv2.drawContours(contour_img, contours, -1, 255, thick)
        self.preprocessed_img = contour_img
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
        self.show_image(contour_img)
        self.status_label.config(text=f"Contours found and thickened (thickness={thick}).")

    def find_and_highlight_mask_candidates(self):
        if self.mask_img is None or self.target_img is None:
            self.status_label.config(text="Images not loaded.")
            return

        # Get threshold from entry
        try:
            self.match_thresh = float(self.threshold_entry.get())
            match_thresh = self.match_thresh
        except ValueError:
            match_thresh = 0.1  # Default threshold for contour matching

        # Find contours in mask image
        ret_mask, mask_bin = cv2.threshold(self.mask_img, 127, 255, cv2.THRESH_BINARY)
        mask_contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not mask_contours:
            self.status_label.config(text="No contours found in mask image.")
            return

        ret_target, target_bin = cv2.threshold(self.target_img, 127, 255, cv2.THRESH_BINARY)
        target_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not target_contours:
            self.status_label.config(text="No contours found in target image.")
            return

        preview_img = cv2.cvtColor(self.target_img.copy(), cv2.COLOR_GRAY2BGR)
        candidate_bboxes = []

        # Get scaled mask size
        mask_h, mask_w = self.mask_img.shape[:2]
        tolerance = 0.2  # 20% tolerance

        for mask_contour in mask_contours:
            for target_contour in target_contours:
                score = cv2.matchShapes(mask_contour, target_contour, cv2.CONTOURS_MATCH_I3, 0.0)
                if score < match_thresh:
                    x, y, w, h = cv2.boundingRect(target_contour)
                    # Filter by size
                    if (
                        abs(w - mask_w) / mask_w < tolerance and
                        abs(h - mask_h) / mask_h < tolerance
                    ):
                        # Fill contour in blue
                        cv2.drawContours(preview_img, [target_contour], -1, (255, 0, 0), thickness=cv2.FILLED)
                        # Outline contour in green
                        cv2.drawContours(preview_img, [target_contour], -1, (0, 255, 0), thickness=2)
                        candidate_bboxes.append((x, y, w, h))

        self.candidate_bboxes = candidate_bboxes
        self.display_img = preview_img
        self.zoom_to_extents()
        match_count = len(candidate_bboxes)
        if match_count == 0:
            self.status_label.config(text=f"No matching contours found (threshold={match_thresh})")
        else:
            self.status_label.config(text=f"Found {match_count} matching contour(s) (threshold={match_thresh}, size tolerance={tolerance*100:.0f}%)")

    def zoom_to_extents(self):
        """Zoom and center the current image to fit the canvas or all matched contours."""
        if self.display_img is None:
            return
        h, w = self.display_img.shape[:2]

        # If candidate bounding boxes are available, zoom to their union
        if hasattr(self, 'candidate_bboxes') and self.candidate_bboxes:
            # Compute union bounding box
            xs = [bbox[0] for bbox in self.candidate_bboxes]
            ys = [bbox[1] for bbox in self.candidate_bboxes]
            ws = [bbox[0] + bbox[2] for bbox in self.candidate_bboxes]
            hs = [bbox[1] + bbox[3] for bbox in self.candidate_bboxes]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(ws), max(hs)
            bw, bh = x_max - x_min, y_max - y_min

            scale_x = self.canvas_width / bw
            scale_y = self.canvas_height / bh
            self.zoom = min(scale_x, scale_y)
            self.offset_x = max(0, int(x_min - (self.canvas_width / self.zoom - bw) / 2))
            self.offset_y = max(0, int(y_min - (self.canvas_height / self.zoom - bh) / 2))
        else:
            fit_zoom = min(self.canvas_width / w, self.canvas_height / h)
            self.zoom = fit_zoom
            self.offset_x = max(0, int((w - self.canvas_width / self.zoom) / 2))
            self.offset_y = max(0, int((h - self.canvas_height / self.zoom) / 2))
        self.show_image(self.display_img)

    # --- Zoom and Pan Methods ---
    def zoom_event(self, event):
        if self.display_img is None:
            return
        h, w = self.display_img.shape[:2]
        fit_zoom = min(self.canvas_width / w, self.canvas_height / h)
        # Determine zoom direction and factor
        if hasattr(event, 'delta'):
            factor = 1.2 if event.delta > 0 else 0.8
        elif hasattr(event, 'num'):
            if event.num == 4:  # Linux scroll up
                factor = 1.2
            elif event.num == 5:  # Linux scroll down
                factor = 0.8
            else:
                return
        else:
            return

        old_zoom = self.zoom
        self.zoom = max(fit_zoom, min(self.zoom * factor, 10.0))

        # Keep mouse position centered
        mouse_x = event.x
        mouse_y = event.y
        rel_x = self.offset_x + mouse_x / old_zoom
        rel_y = self.offset_y + mouse_y / old_zoom
        self.offset_x = max(0, min(int(rel_x - mouse_x / self.zoom), w - int(self.canvas_width / self.zoom)))
        self.offset_y = max(0, min(int(rel_y - mouse_y / self.zoom), h - int(self.canvas_height / self.zoom)))
        self.show_image(self.display_img)

    def start_pan(self, event):
        self.last_mouse_pos = (event.x, event.y)

    def pan_event(self, event):
        if self.last_mouse_pos and self.display_img is not None:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            h, w = self.display_img.shape[:2]
            self.offset_x = max(0, min(self.offset_x - int(dx / self.zoom), w - int(self.canvas_width / self.zoom)))
            self.offset_y = max(0, min(self.offset_y - int(dy / self.zoom), h - int(self.canvas_height / self.zoom)))
            self.last_mouse_pos = (event.x, event.y)
            self.show_image(self.display_img)

    def end_pan(self, event):
        self.last_mouse_pos = None

    # Optional: Shift+Left for pan (for touchpads)
    def start_pan_shift(self, event):
        if event.state & 0x0001:  # Shift key
            self.last_mouse_pos = (event.x, event.y)

    def pan_event_shift(self, event):
        if self.last_mouse_pos and self.display_img is not None:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            h, w = self.display_img.shape[:2]
            self.offset_x = max(0, min(self.offset_x - int(dx / self.zoom), w - int(self.canvas_width / self.zoom)))
            self.offset_y = max(0, min(self.offset_y - int(dy / self.zoom), h - int(self.canvas_height / self.zoom)))
            self.last_mouse_pos = (event.x, event.y)
            self.show_image(self.display_img)

    def end_pan_shift(self, event):
        self.last_mouse_pos = None

    def overlay_mask_bottom_right(self):
        if self.target_img is None or self.mask_img is None:
            return

        # Ensure both images are grayscale and mask is scaled
        mask_h, mask_w = self.mask_img.shape[:2]
        target_h, target_w = self.target_img.shape[:2]

        # Prepare color preview of target
        preview_img = cv2.cvtColor(self.target_img.copy(), cv2.COLOR_GRAY2BGR)

        # Overlay mask in bottom right corner
        y_offset = target_h - mask_h
        x_offset = target_w - mask_w

        if y_offset < 0 or x_offset < 0:
            self.status_label.config(text="Mask is larger than target image. Cannot overlay.")
            self.display_img = self.mask_img
            return

        # Create colored mask (red channel)
        colored_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        colored_mask[:, :, 2] = self.mask_img  # Red channel

        # Blend mask with target image (alpha blending)
        alpha = 0.5
        roi = preview_img[y_offset:y_offset+mask_h, x_offset:x_offset+mask_w]
        blended = cv2.addWeighted(roi, 1-alpha, colored_mask, alpha, 0)
        preview_img[y_offset:y_offset+mask_h, x_offset:x_offset+mask_w] = blended

        self.display_img = preview_img
        self.zoom_to_extents()
        self.status_label.config(text="Mask overlaid in bottom right corner of target image.")

def get_pixel_size(path):
    with rasterio.open(path) as src:
        # src.transform is an Affine object: (a, b, c, d, e, f)
        # a = pixel width, e = pixel height (usually negative)
           pixel_width = src.transform.a
           pixel_height = abs(src.transform.e)
           return pixel_width, pixel_height

def scale_mask_to_target(mask_path, target_path):
    mask_pixel_size = get_pixel_size(mask_path)
    target_pixel_size = get_pixel_size(target_path)

    print(f"Mask pixel scale before scaling: {mask_pixel_size}")
    print(f"Target pixel scale: {target_pixel_size}")

    scale_x = mask_pixel_size[0] / target_pixel_size[0]
    scale_y = mask_pixel_size[1] / target_pixel_size[1]

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    new_size = (
        int(mask_img.shape[1] * scale_x),
        int(mask_img.shape[0] * scale_y)
    )
    scaled_mask = cv2.resize(mask_img, new_size, interpolation=cv2.INTER_NEAREST)

    # Print pixel scale after conversion (now matches target)
    print(f"Mask pixel scale after scaling: {target_pixel_size}")
    return scaled_mask

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskFinderGUI(root)
    root.mainloop()

