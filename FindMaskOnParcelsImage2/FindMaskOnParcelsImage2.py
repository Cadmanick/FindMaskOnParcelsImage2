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
        self.display_img = None

        # Pan and zoom state
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_mouse_pos = None

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

        # --- Button row frame, left justified ---
        button_row = tk.Frame(root)
        button_row.pack(side=tk.TOP, pady=(10, 0), anchor="w", fill=tk.X)

        # "1-Load Target Image" button
        self.load_target_btn = Button(button_row, text="1-Load Target Image", command=self.load_target_image)
        self.load_target_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Mask rotation field and label (vertical in its own frame)
        mask_rot_frame = tk.Frame(button_row)
        mask_rot_frame.pack(side=tk.LEFT, padx=5)
        Label(mask_rot_frame, text="Mask Rotation (deg.)").pack(side=tk.TOP, anchor="w")
        self.mask_rotation_entry = Entry(mask_rot_frame, width=10)
        self.mask_rotation_entry.insert(0, "16")
        self.mask_rotation_entry.pack(side=tk.TOP, anchor="w", pady=(2, 6))

        # Similarity threshold field and label (vertical in its own frame, right side)
        sim_thresh_frame = tk.Frame(button_row)
        sim_thresh_frame.pack(side=tk.RIGHT, padx=5)
        Label(sim_thresh_frame, text="Similarity Threshold").pack(side=tk.TOP, anchor="e")
        self.similarity_threshold_entry = Entry(sim_thresh_frame, width=10)
        self.similarity_threshold_entry.insert(0, "0.05")
        self.similarity_threshold_entry.pack(side=tk.TOP, anchor="e", pady=(2, 6))

        # "2-Load Mask Image" button
        self.load_mask_btn = Button(button_row, text="2-Load Mask Image", command=self.load_mask_image)
        self.load_mask_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = Label(root, text="Load images to begin.")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind zoom and pan events
        self.canvas.bind("<MouseWheel>", self.zoom_event)  # Windows
        self.canvas.bind("<Button-4>", self.zoom_event)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom_event)    # Linux scroll down
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_event)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        self.canvas.bind("<ButtonPress-1>", self.start_pan_shift)
        self.canvas.bind("<B1-Motion>", self.pan_event_shift)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan_shift)

        # Add MaskFinderTools button to the button_row
        self.tools = MaskFinderTools(self, button_row)

    def show_image(self, img):
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

        # --- Overlay mask in bottom right of viewport, scaled by zoom ---
        if self.mask_img is not None:
            # The mask should be scaled by the zoom so its pixel size matches the target image in the viewport
            mask_h, mask_w = self.mask_img.shape[:2]
            # Calculate overlay size in viewport: scale mask by current zoom
            overlay_h = int(mask_h * self.zoom)
            overlay_w = int(mask_w * self.zoom)
            # Prevent overlay from exceeding viewport
            overlay_h = min(overlay_h, self.canvas_height)
            overlay_w = min(overlay_w, self.canvas_width)
            mask_resized = cv2.resize(self.mask_img, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
            colored_mask[:, :, 2] = mask_resized
            alpha = 0.5
            alpha_mask = (mask_resized.astype(np.float32) / 255.0) * alpha
            alpha_mask = np.expand_dims(alpha_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2)
            y_offset = self.canvas_height - overlay_h
            x_offset = self.canvas_width - overlay_w
            if resized.ndim == 2:
                resized_color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            else:
                resized_color = resized.copy()
            roi = resized_color[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w].astype(np.float32)
            mask_foreground = (mask_resized > 0).astype(np.float32)
            mask_foreground = np.expand_dims(mask_foreground, axis=2)
            mask_foreground = np.repeat(mask_foreground, 3, axis=2)
            blended = roi * (1 - alpha_mask * mask_foreground) + colored_mask.astype(np.float32) * (alpha_mask * mask_foreground)
            blended = blended.astype(np.uint8)
            resized_color[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = blended
            preview_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        else:
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

        # --- Get rotation value ---
        try:
            rotation_deg = float(self.mask_rotation_entry.get())
        except Exception:
            rotation_deg = 16.0  # fallback default

        # Wait until target image is loaded
        if self.target_img is not None:
            target_path = self.target_path_entry.get()
            mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                self.status_label.config(text="Failed to load mask image.")
                return
            mask_img = self.rotate_image(mask_img, rotation_deg)
            mask_img = self.scale_mask_to_target_array(mask_img, target_path)
            self.mask_img = mask_img
            self.zoom_to_extents()
        else:
            mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_img = self.rotate_image(mask_img, rotation_deg)
                self.mask_img = mask_img
                self.status_label.config(text=f"Loaded mask image: {file_path}")
                self.zoom_to_extents()
            else:
                self.status_label.config(text="Failed to load mask image.")

    # --- Helper to rotate image ---
    def rotate_image(self, img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated

    # --- Helper to scale mask array to target ---
    def scale_mask_to_target_array(self, mask_img, target_path):
        mask_pixel_size = get_pixel_size(self.mask_path_entry.get())
        target_pixel_size = get_pixel_size(target_path)
        scale_x = mask_pixel_size[0] / target_pixel_size[0]
        scale_y = mask_pixel_size[1] / target_pixel_size[1]
        new_size = (
            int(mask_img.shape[1] * scale_x),
            int(mask_img.shape[0] * scale_y)
        )
        scaled_mask = cv2.resize(mask_img, new_size, interpolation=cv2.INTER_NEAREST)
        print(f"Mask pixel scale after scaling: {target_pixel_size}")
        return scaled_mask

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

    def zoom_event(self, event):
        if self.display_img is None:
            return
        h, w = self.display_img.shape[:2]
        fit_zoom = min(self.canvas_width / w, self.canvas_height / h)
        if hasattr(event, 'delta'):
            factor = 1.2 if event.delta > 0 else 0.8
        elif hasattr(event, 'num'):
            if event.num == 4:
                factor = 1.2
            elif event.num == 5:
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

    def zoom_to_extents(self):
        """Zoom and center the current image to fit the canvas."""
        if self.display_img is None:
            return
        h, w = self.display_img.shape[:2]
        fit_zoom = min(self.canvas_width / w, self.canvas_height / h)
        self.zoom = fit_zoom
        self.offset_x = max(0, int((w - self.canvas_width / self.zoom) / 2))
        self.offset_y = max(0, int((h - self.canvas_height / self.zoom) / 2))
        img_with_mask = self.overlay_mask_bottom_right_with_transparency(self.display_img, self.mask_img)
        self.show_image(img_with_mask)

    def overlay_mask_bottom_right_with_transparency(self, viewport_img, mask_img, alpha=0.5):
        # """
        # Overlay mask_img in the bottom right corner of the viewport_img,
        # scaling the mask according to the target pixel scale, and keeping it in view.
        # Only non-black mask pixels are shown as 50% transparent red.
        # """
        if viewport_img is None or mask_img is None:
            return viewport_img

        # Scale mask to fit the target pixel scale (already done if mask_img is scaled)
        mask_h, mask_w = mask_img.shape[:2]
        canvas_h, canvas_w = self.canvas_height, self.canvas_width

        # Optionally, scale mask by zoom (or keep a fixed pixel size)
        scaled_mask_h = int(mask_h * self.zoom)
        scaled_mask_w = int(mask_w * self.zoom)
        overlay_h = min(scaled_mask_h, canvas_h)
        overlay_w = min(scaled_mask_w, canvas_w)

        # Resize mask to overlay size
        mask_resized = cv2.resize(mask_img, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
        # Only non-black pixels are shown as red
        colored_mask[:, :, 2] = mask_resized

        # Prepare alpha mask (per-pixel mask * global alpha)
        alpha_mask = (mask_resized.astype(np.float32) / 255.0) * alpha
        alpha_mask = np.expand_dims(alpha_mask, axis=2)  # (h, w, 1)
        alpha_mask = np.repeat(alpha_mask, 3, axis=2)    # (h, w, 3)

        # Overlay in bottom right of viewport
        y_offset = canvas_h - overlay_h
        x_offset = canvas_w - overlay_w

        # Ensure viewport_img is color
        if viewport_img.ndim == 2:
            viewport_img_color = cv2.cvtColor(viewport_img, cv2.COLOR_GRAY2BGR)
        else:
            viewport_img_color = viewport_img.copy()

        # Get ROI from viewport
        roi = viewport_img_color[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w].astype(np.float32)

        # Blend only where mask is nonzero (foreground)
        mask_foreground = (mask_resized > 0).astype(np.float32)
        mask_foreground = np.expand_dims(mask_foreground, axis=2)
        mask_foreground = np.repeat(mask_foreground, 3, axis=2)
        blended = roi * (1 - alpha_mask * mask_foreground) + colored_mask.astype(np.float32) * (alpha_mask * mask_foreground)
        blended = blended.astype(np.uint8)

        # Place blended ROI back into viewport
        result_img = viewport_img_color.copy()
        result_img[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = blended
        return result_img

class MaskFinderTools:
    def __init__(self, gui, parent_frame):
        self.gui = gui
        self.button = Button(parent_frame, text="Find Contours (Green Overlay)", command=self.find_and_overlay_contours)
        self.button.pack(side=tk.LEFT, padx=5, pady=5)

    def find_compound_contours(self, img, mask_img=None, area_tolerance=0.314, abut_distance=1):
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        # Group abutting contours (greedy grouping)
        groups = []
        used = set()
        for i, rect1 in enumerate(bounding_boxes):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j, rect2 in enumerate(bounding_boxes):
                if j == i or j in used:
                    continue
                if self._rects_abut(rect1, rect2, abut_distance):
                    group.append(j)
                    used.add(j)
            groups.append(group)

        # Combine points for each group and compute area
        group_areas = []
        group_contours = []
        for group in groups:
            pts = np.vstack([contours[idx] for idx in group])
            area = cv2.contourArea(pts)
            group_areas.append(area)
            group_contours.append([contours[idx] for idx in group])

        # Get mask area
        if mask_img is None:
            mask_img = getattr(self.gui, "mask_img", None)
        mask_area = 0
        if mask_img is not None:
            mask_area = np.count_nonzero(mask_img)
        else:
            mask_area = max(group_areas) if group_areas else 0

        # Filter by area tolerance
        min_area = mask_area * (1 - area_tolerance)
        max_area = mask_area * (1 + area_tolerance)
        filtered_contours = [group for group, area in zip(group_contours, group_areas) if min_area <= area <= max_area]
        # For visualization, compute bounding boxes of filtered groups
        filtered_boxes = []
        for group in filtered_contours:
            pts = np.vstack(group)
            x, y, w, h = cv2.boundingRect(pts)
            filtered_boxes.append((x, y, w, h))
        return filtered_contours, filtered_boxes  # Note: filtered_contours is now a list of lists

    def _rects_abut(self, rect1, rect2, abut_distance):
        # rect: (x, y, w, h)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        # Check if rectangles touch or are within abut_distance
        if (x1 + w1 + abut_distance >= x2 and x2 + w2 + abut_distance >= x1 and
            y1 + h1 + abut_distance >= y2 and y2 + h2 + abut_distance >= y1):
            return True
        return False

    def find_features(self, compound_groups, method=cv2.CONTOURS_MATCH_I1, parameter=0.0, similarity_threshold=0.1):
        """
        Uses cv2.matchShapes to find compound contours similar to the reference compound contour.
        Returns a list of matching compound contours.
        """
        if not compound_groups:
            return []
        # Combine each group into a single contour
        combined_contours = [np.vstack(group) for group in compound_groups]
        # Use the largest combined contour as reference
        ref_contour = max(combined_contours, key=cv2.contourArea)
        matches = []
        for cnt in combined_contours:
            score = cv2.matchShapes(ref_contour, cnt, method, parameter)
            if score < similarity_threshold:
                matches.append(cnt)
        return matches

    def find_and_overlay_contours(self):
        if self.gui.target_img is None:
            self.gui.status_label.config(text="Target image not loaded.")
            return

        img = self.gui.target_img.copy()
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Get compound contour groups and bounding boxes
        compound_groups, bounding_boxes = self.find_compound_contours(img, mask_img=self.gui.mask_img)

        # Draw filled green compound contours (overlay style)
        contour_layer = np.zeros_like(overlay)
        for group in compound_groups:
            cv2.drawContours(contour_layer, [np.vstack(group)], -1, (0, 255, 0), thickness=cv2.FILLED)
        # Draw bounding boxes in red
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(contour_layer, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        # --- Get similarity threshold from GUI ---
        try:
            similarity_threshold = float(self.gui.similarity_threshold_entry.get())
        except Exception:
            similarity_threshold = 0.05  # fallback default

        # --- Find and overlay similar features using matchShapes ---
        matching_contours = self.find_features(
            compound_groups,
            method=cv2.CONTOURS_MATCH_I3,
            parameter=0.0,
            similarity_threshold=similarity_threshold
        )
        for cnt in matching_contours:
            cv2.drawContours(contour_layer, [cnt], -1, (255, 0, 0), thickness=2)  # Blue for matches

        # Blend contour layer with original (50% transparency)
        blended = cv2.addWeighted(overlay, 0.5, contour_layer, 0.5, 0)

        self.gui.display_img = blended
        self.gui.show_image(blended)
        self.gui.status_label.config(
            text=f"Compound contours found: {len(bounding_boxes)} (green overlay, red bounding boxes, blue features, 50% transparent, similarity threshold {similarity_threshold})"
        )





if __name__ == "__main__":
    root = tk.Tk()
    app = MaskFinderGUI(root)
    root.mainloop()
