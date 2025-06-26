#!/usr/bin/env python3
"""
SaulsART Paint-By-Number Generator 
==================================

Turn any photo into a printable paint-by-number 

What it does
------------
- Automatic colour-pallette reduction with OpenCV K-means
- Creates single-pixel-wide outlines
- Places one, size aware number inside every region corresponding to colour
- Generates four assets per run:
    1. '*_template.png'     - black-outline drawing (no numbers - ideal for tracing)
    2. '*_template.svg'     - outline drawing plus red numbers for easy visibility (4-10 pt)
    3. '*_colours.png'      - colour guide with swatches + mixing guide (HEX codes)
    4. '*_reference.png'    - fully coloured preview reference image

Requirements:
    pip install opencv-python pillow numpy scipy scikit-learn

Running the GUI:
    # Open the app window 
    python saulsart_pbn_generator.py 

Author: Erika Saul - MIT Licence
"""

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:         # head-less env (Streamlit Cloud)
    tk = None      
    filedialog = messagebox = tkk = None                 # dummy placeholder so the rest of the file imports
import os
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from scipy import ndimage
from xml.etree.ElementTree import Element, SubElement, ElementTree
from sklearn.cluster import KMeans
import threading
import sys
import argparse
import pathlib
import types
# Wipe Jupyter lags if we are inside a notebook
if "ipykernel_launcher" in sys.argv[0]:
    sys.argv = ["saulsart_pbn_generator.py"]

# CLI arguments
parser = argparse.ArgumentParser(description="Fast Paint-by-Numbers Generator")
parser.add_argument('-c', '--colors', type=int, default=16, help="Number of colors")
parser.add_argument('--min-area', type=int, default=50, help="Minimum region size")
parser.add_argument('--no-svg', action='store_true', help="Skip SVG output")
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

QUALITY_PRESETS = {
    "fast": {
        "n_colors": 5,
        "min_area": 500,
        "smoothing": 5,
        "description": "Fast (5 colors)"
    },
    "standard": {
        "n_colors": 10,
        "min_area": 200,
        "smoothing": 5,
        "description": "Standard (10 colors)"
    },
    "detailed": {
        "n_colors": 16,
        "min_area": 100,
        "smoothing": 3,
        "description": "Detailed (16 colors)"
    },
    "ultra": {
        "n_colors": 30,
        "min_area": 50,
        "smoothing": 3,
        "description": "Ultra (30 colors)"
    }
}

SIZE_PRESETS = {
    "small": {"width": 800, "height": 1100, "description": "Small (A4)"},
    "medium": {"width": 1200, "height": 1600, "description": "Medium"},
    "large": {"width": 1600, "height": 2000, "description": "Large"},
}

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class FastPaintByNumbersApp(tk.Tk):
    """Fast paint-by-numbers generator application"""
    
    def __init__(self):
        super().__init__()
        self.title("🎨 Fast Paint-by-Numbers Generator V5")
        self.geometry("750x700")
        self.configure(bg='#f0f0f0')
        
        self.setup_variables()
        self.setup_ui()
        self.check_dependencies()
    
    def setup_variables(self):
        self.image_path = ""
        self.output_dir = "paint_by_numbers_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.quality_var = tk.StringVar(value="standard")
        self.size_var = tk.StringVar(value="medium")
        self.path_var = tk.StringVar(value="No image selected")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()
        self.colors_var = tk.IntVar(value=16)
        self.detail_var = tk.IntVar(value=100)
        self.number_color_var = tk.StringVar(value="red")
        self.font_adjust_var = tk.IntVar(value=0)  # Font size adjustment
        
        self.processing_thread = None
        self.cancel_processing = False
    
    def setup_ui(self):
        main_frame = tk.Frame(self, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        tk.Label(main_frame, text="🎨 Fast Paint-by-Numbers Generator V5",
                font=('Arial', 18, 'bold'), bg='#f0f0f0').pack(pady=(0, 20))
        
        # Image selection
        img_frame = tk.LabelFrame(main_frame, text="📷 Image", font=('Arial', 12, 'bold'),
                                 bg='white', relief='raised', bd=2)
        img_frame.pack(fill='x', pady=(0, 15))
        
        tk.Button(img_frame, text="Select Image", command=self.browse_image,
                 bg='#2196F3', fg='white', font=('Arial', 11, 'bold'),
                 relief='flat', padx=20, pady=8).pack(pady=10)
        
        tk.Label(img_frame, textvariable=self.path_var, bg='white',
                font=('Arial', 9), fg='#666').pack(pady=(0, 10))
        
        # Settings
        settings_frame = tk.LabelFrame(main_frame, text="⚙️ Settings", 
                                      font=('Arial', 12, 'bold'),
                                      bg='white', relief='raised', bd=2)
        settings_frame.pack(fill='x', pady=(0, 15))
        
        settings_inner = tk.Frame(settings_frame, bg='white')
        settings_inner.pack(padx=15, pady=10)
        
        # Quality preset
        # tk.Label(settings_inner, text="Quality:", font=('Arial', 10, 'bold'),
                # bg='white').grid(row=0, column=0, sticky='w', padx=5)
        # ttk.Combobox(settings_inner, textvariable=self.quality_var,
                    # values=[f"{k}: {v['description']}" for k, v in QUALITY_PRESETS.items()],
                    # state="readonly", width=25).grid(row=0, column=1, padx=5)
        
        # Size
        # tk.Label(settings_inner, text="Size:", font=('Arial', 10, 'bold'),
                # bg='white').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        # ttk.Combobox(settings_inner, textvariable=self.size_var,
                    # values=[f"{k}: {v['description']}" for k, v in SIZE_PRESETS.items()],
                    # state="readonly", width=25).grid(row=1, column=1, padx=5, pady=5)
        
        # Colors
        tk.Label(settings_inner, text="Colors:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=0, column=0, sticky='w', padx=5)
        tk.Scale(settings_inner, from_=2, to=50, orient='horizontal',
                variable=self.colors_var, length=200, bg='white').grid(row=0, column=1, padx=5)
        
        # Min area
        tk.Label(settings_inner, text="Min Area:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=1, column=0, sticky='w', padx=5)
        tk.Scale(settings_inner, from_=20, to=1000, orient='horizontal',
                variable=self.detail_var, length=200, bg='white').grid(row=1, column=1, padx=5)
        
        # Number color
        tk.Label(settings_inner, text="Number Color:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        color_frame = tk.Frame(settings_inner, bg='white')
        color_frame.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        for color in ["red", "black", "blue", "green"]:
            tk.Radiobutton(color_frame, text=color.capitalize(), variable=self.number_color_var,
                          value=color, bg='white').pack(side='left', padx=5)
        
        # Font size adjustment
        tk.Label(settings_inner, text="Font Size:", font=('Arial', 10, 'bold'),
                bg='white').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        font_frame = tk.Frame(settings_inner, bg='white')
        font_frame.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        tk.Scale(font_frame, from_=-2, to=2, orient='horizontal',
                variable=self.font_adjust_var, length=150, bg='white',
                showvalue=False).pack(side='left')
        tk.Label(font_frame, text="smaller ← → larger", font=('Arial', 8),
                bg='white', fg='#666').pack(side='left', padx=5)
        
        # Process button
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.generate_btn = tk.Button(button_frame, text="🚀 Generate",
                                     command=self.process_image,
                                     bg='#4CAF50', fg='white',
                                     font=('Arial', 14, 'bold'),
                                     relief='flat', padx=30, pady=12)
        self.generate_btn.pack(side='left', padx=5)
        
        self.cancel_btn = tk.Button(button_frame, text="Cancel",
                                   command=self.cancel_processing,
                                   bg='#F44336', fg='white',
                                   font=('Arial', 12, 'bold'),
                                   relief='flat', padx=20, pady=10)
        
        # Progress
        progress_frame = tk.LabelFrame(main_frame, text="📊 Progress",
                                      font=('Arial', 12, 'bold'),
                                      bg='white', relief='raised', bd=2)
        progress_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        ttk.Progressbar(progress_frame, variable=self.progress_var,
                       maximum=100).pack(fill='x', padx=15, pady=(10, 5))
        
        self.log_text = tk.Text(progress_frame, height=8, wrap='word',
                               font=('Consolas', 9), bg='#f8f8f8')
        self.log_text.pack(fill='both', expand=True, padx=15, pady=(5, 10))
        
        # Status bar
        tk.Label(main_frame, textvariable=self.status_var, 
                font=('Arial', 9), bg='#e0e0e0',
                anchor='w').pack(fill='x')
    
    def check_dependencies(self):
        try:
            import cv2
            import sklearn
            self.log_message("✅ Dependencies OK")
            self.log_message("🎨 Ready to generate paint-by-numbers!")
            self.log_message("📌 Numbers: tiny red text (4-10pt)")
            self.log_message("💡 Use font slider if numbers don't fit")
        except ImportError:
            self.log_message("❌ Missing dependencies")
            self.generate_btn.config(state='disabled')
    
    def log_message(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {msg}\n")
        self.log_text.see('end')
        self.update_idletasks()
    
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.path_var.set(f"Selected: {os.path.basename(file_path)}")
            self.log_message(f"📷 Loaded: {os.path.basename(file_path)}")
    
    def cancel_processing(self):
        self.cancel_processing = True
        self.cancel_btn.pack_forget()
    
    def process_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first")
            return
        
        self.generate_btn.config(state='disabled')
        self.cancel_btn.pack(side='left', padx=5)
        
        self.processing_thread = threading.Thread(target=self._process_thread, daemon=True)
        self.processing_thread.start()
    
    def _process_thread(self):
        try:
            self.cancel_processing = False
            self.log_message("\n" + "="*50)
            self.log_message("🚀 STARTING FAST PAINT-BY-NUMBERS")
            
            # Get settings
            quality_key = self.quality_var.get().split(':')[0]
            size_key = self.size_var.get().split(':')[0]
            
            config = QUALITY_PRESETS[quality_key].copy()
            config['n_colors'] = self.colors_var.get()
            config['min_area'] = self.detail_var.get()
            config['number_color'] = self.number_color_var.get()
            config['font_adjust'] = self.font_adjust_var.get()  # Add font adjustment
            
            size_config = SIZE_PRESETS[size_key]
            
            # Process
            processor = FastProcessor(self, config, size_config)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            processor.generate(self.image_path, base_name)
            
            if not self.cancel_processing:
                self.log_message("✅ COMPLETE!")
                messagebox.showinfo("Success", 
                    f"Paint-by-numbers generated!\n\nOutput: {self.output_dir}/{base_name}/")
            
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        
        finally:
            self.generate_btn.config(state='normal')
            self.cancel_btn.pack_forget()
            self.progress_var.set(0)

# =============================================================================
# FAST PROCESSING ENGINE
# =============================================================================

class FastProcessor:
    """Optimized processing engine"""
    
    def __init__(self, app, config, size_config):
        self.app = app
        self.config = config
        self.size_config = size_config
        self.colors = []
        self.regions = []
    
    def update_progress(self, value, msg=""):
        self.app.progress_var.set(value)
        if msg:
            self.app.log_message(msg)
        self.app.update_idletasks()
    
    def generate(self, image_path, base_name):
        """Generate paint-by-numbers - FAST version"""
        
        # 1. Load image
        self.update_progress(10, "📷 Loading image...")
        img = self.load_image(image_path)
        
        # 2. Color quantization
        self.update_progress(20, "🎨 Quantizing colors...")
        quantized, labels = self.fast_quantize(img)
        
        # 3. Create regions
        self.update_progress(40, "🔧 Creating regions...")
        region_map = self.fast_create_regions(labels)
        
        # 4. Extract boundaries
        self.update_progress(60, "📐 Extracting boundaries...")
        boundaries = self.fast_extract_boundaries(region_map)
        
        # 5. Find number positions
        self.update_progress(70, "🔍 Finding number positions...")
        self.find_all_number_positions(region_map)
        
        # 6. Generate outputs
        self.update_progress(80, "💾 Generating outputs...")
        self.generate_outputs(region_map, boundaries, base_name)
        
        self.update_progress(100, "✅ Complete!")
    
    def load_image(self, path):
        """Load and resize image"""
        img = Image.open(path).convert('RGB')
        
        # Resize to target
        target_w = self.size_config['width']
        target_h = self.size_config['height']
        img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        
        self.app.log_message(f"📐 Size: {img.size}")
        return np.array(img)
    
    def fast_quantize(self, img):
        """Fast color quantization"""
        h, w = img.shape[:2]
        n_colors = self.config['n_colors']
        
        # Reshape and cluster
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Fast K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        # Create color palette
        self.colors = []
        for i, center in enumerate(centers):
            self.colors.append({
                'idx': i + 1,
                'rgb': tuple(map(int, center)),
                'hex': '#{:02X}{:02X}{:02X}'.format(*map(int, center))
            })
        
        # Apply quantization
        quantized = centers[labels.flatten()].reshape(h, w, 3).astype(np.uint8)
        labels = labels.reshape(h, w)
        
        self.app.log_message(f"✅ {n_colors} colors created")
        return quantized, labels
    
    def fast_create_regions(self, labels):
        """Fast region creation with area filtering"""
        h, w = labels.shape
        region_map = labels + 1  # 1-indexed
        
        # Quick smoothing
        smoothing = self.config.get('smoothing', 5)
        if smoothing > 0:
            if smoothing % 2 == 0:
                smoothing += 1
            region_map = cv2.medianBlur(region_map.astype(np.uint8), smoothing)
        
        # Fast small region removal using connected components
        min_area = self.config['min_area']
        self.app.log_message(f"   Removing regions < {min_area}px...")
        
        # Process all regions at once
        cleaned = np.zeros_like(region_map)
        
        for color_idx in range(1, len(self.colors) + 1):
            if self.app.cancel_processing:
                break
                
            mask = (region_map == color_idx).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            
            # Find components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Keep only large components
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == i] = color_idx
            
        # Fill small holes by assigning to nearest neighbor
        unassigned = (cleaned == 0)
        if np.any(unassigned):
            # Use distance transform to find nearest assigned pixel
            assigned = ~unassigned
            dist, indices = ndimage.distance_transform_edt(unassigned, return_indices=True)
            cleaned[unassigned] = cleaned[indices[0][unassigned], indices[1][unassigned]]
        
        self.app.log_message("✅ Regions created")
        return cleaned.astype(np.int32)
    
    def fast_extract_boundaries(self, region_map):
        """Extract clean boundaries between regions"""
        h, w = region_map.shape
        
        # Simple edge detection - find where regions change
        # Horizontal edges
        h_edges = np.zeros((h, w), dtype=bool)
        h_edges[:, :-1] = region_map[:, :-1] != region_map[:, 1:]
        
        # Vertical edges
        v_edges = np.zeros((h, w), dtype=bool)
        v_edges[:-1, :] = region_map[:-1, :] != region_map[1:, :]
        
        # Combine
        boundaries = h_edges | v_edges
        
        self.app.log_message("✅ Boundaries extracted")
        return boundaries.astype(np.uint8) * 255
    
    def find_all_number_positions(self, region_map):
        """
        Find one, and only one, number position for every connected
        component in the region map.  Uses a distance-transform “deep-inside”
        pixel instead of the raw centroid, so the label is always inside
        the correct area, and never overlaps another label.
        """
        self.regions = []
        occupied = set()                # keep track of used (x, y) pixels

        for color_idx in range(1, len(self.colors) + 1):
            mask = (region_map == color_idx).astype(np.uint8)
            if mask.sum() == 0:
                continue

            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

            for i in range(1, n):       # skip label 0 (= background)
                area   = stats[i, cv2.CC_STAT_AREA]
                width  = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                if area < 10:           # ignore specks too small to paint
                    continue

                component = (labels == i).astype(np.uint8)

                # Pick a pixel WELL INSIDE the component
                if area >= 30:
                    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
                    y, x = np.unravel_index(dist.argmax(), dist.shape)
                else:                   # tiny blob: fall back to centroid
                    x, y = map(int, np.round(centroids[i]))

                # Guarantee uniqueness – nudge right until the slot is free
                while (x, y) in occupied:
                    x += 1

                occupied.add((x, y))

                self.regions.append({
                    "color_idx": color_idx,
                    "position" : (x, y),
                    "area"     : area,
                    "width"    : width,
                    "height"   : height
                })

        self.app.log_message(f"✅ Found {len(self.regions)} regions to number (no overlaps)")

    
    def generate_outputs(self, region_map, boundaries, base_name):
        """Generate output files"""
        output_dir = os.path.join(self.app.output_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. PNG template
        png_path = os.path.join(output_dir, f"{base_name}_template.png")
        self.create_png_template(region_map, boundaries, png_path)
        
        # 2. SVG (optional)
        if not args.no_svg:
            svg_path = os.path.join(output_dir, f"{base_name}_template.svg")
            self.create_svg_template(region_map, boundaries, svg_path)
        
        # 3. Color guide
        guide_path = os.path.join(output_dir, f"{base_name}_colors.png")
        self.create_color_guide(guide_path)
        
        # 4. Reference
        ref_path = os.path.join(output_dir, f"{base_name}_reference.png")
        self.create_reference(region_map, ref_path)
    
    def create_png_template(self, region_map, boundaries, output_path):
        """Create PNG template *without* numbers – outlines only."""
        h, w = region_map.shape

        # 1. White canvas with black boundaries
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        template[boundaries > 0] = [0, 0, 0]

        # 2. Save
        Image.fromarray(template).save(output_path, "PNG", quality=95)
        self.app.log_message("✅ PNG template (no numbers) saved")

    
    def create_svg_template(self, region_map, boundaries, output_path):
        """Create SVG template"""
        h, w = region_map.shape
        
        svg = Element("svg", {
            "width": str(w),
            "height": str(h),
            "viewBox": f"0 0 {w} {h}",
            "xmlns": "http://www.w3.org/2000/svg"
        })
        
        # White background
        SubElement(svg, "rect", {
            "width": str(w), "height": str(h),
            "fill": "white"
        })
        
        # Extract boundary paths efficiently
        # Convert boundaries to contours
        contours, _ = cv2.findContours(boundaries, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundary paths
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Create path
            points = " ".join([f"{p[0][0]},{p[0][1]}" for p in contour])
            SubElement(svg, "polyline", {
                "points": points,
                "fill": "none",
                "stroke": "black",
                "stroke-width": "1"
            })
        
        # Add numbers (no circles, no background, EVEN SMALLER)
        font_adjust = self.config.get('font_adjust', 0)
        
        for region in self.regions:
            cx, cy = region['position']
            color_num = region['color_idx']
            
            # Font size based on area and number of digits
            # Use smaller font for double-digit numbers
            is_double_digit = color_num >= 10
            
            if region['area'] < 50:
                base_size = 4
            elif region['area'] < 100:
                base_size = 4 if is_double_digit else 5
            elif region['area'] < 300:
                base_size = 5 if is_double_digit else 6
            elif region['area'] < 1000:
                base_size = 6 if is_double_digit else 8
            else:
                base_size = 8 if is_double_digit else 10
            
            # Apply font adjustment
            font_size = str(max(3, base_size + font_adjust))
            
            # Add text directly in configured color
            number_color = self.config.get('number_color', 'red')
            text = SubElement(svg, "text", {
                "x": str(cx),
                "y": str(cy),
                "text-anchor": "middle",
                "dominant-baseline": "central",
                "font-family": "Arial",
                "font-size": font_size,
                "fill": number_color
            })
            text.text = str(color_num)
        
        # Save
        tree = ElementTree(svg)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        self.app.log_message("✅ SVG template saved")
    
    def create_color_guide(self, output_path):
        """Create color guide"""
        width = 300
        swatch_size = 40
        padding = 20
        height = padding * 2 + len(self.colors) * (swatch_size + 10)
        
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Title
        draw.text((padding, padding), "Color Guide", fill="black", font=font)
        
        # Colors
        y = padding + 30
        for color in self.colors:
            # Swatch
            draw.rectangle([padding, y, padding + swatch_size, y + swatch_size],
                          fill=color['rgb'], outline="black")
            
            # Number and hex
            draw.text((padding + swatch_size + 10, y + swatch_size//2),
                     f"{color['idx']}: {color['hex']}", 
                     fill="black", font=font, anchor="lm")
            
            y += swatch_size + 10
        
        img.save(output_path, "PNG")
        self.app.log_message("✅ Color guide saved")
    
    def create_reference(self, region_map, output_path):
        """Create colored reference"""
        h, w = region_map.shape
        reference = np.zeros((h, w, 3), dtype=np.uint8)
        
        for color in self.colors:
            mask = (region_map == color['idx'])
            reference[mask] = color['rgb']
        
        Image.fromarray(reference).save(output_path, "PNG")
        self.app.log_message("✅ Reference saved")

# ---------------------------------------------------------------------------
# 🖥️  HEADLESS ENGINE — callable from Streamlit or tests, no Tkinter needed
# ---------------------------------------------------------------------------
def generate_pbn(
    img_path: str,
    n_colors: int,
    min_area: int,
    number_color: str,
    font_adjust: int,
    make_svg: bool,
    output_root: str = "web_output"
) -> pathlib.Path:
    """
    Run FastProcessor without starting the Tk GUI.
    Returns the path to the output directory that holds PNG/SVG files.
    """
    import shutil, itertools, tempfile   # local imports to keep globals clean

    # Build minimal configs -------------------------------------------------
    cfg = {
        "n_colors": n_colors,
        "min_area": min_area,
        "smoothing": 3,
        "number_color": number_color,
        "font_adjust": font_adjust,
    }
    size_cfg = {"width": 1600, "height": 2000}

    # Dummy object supplying just what FastProcessor expects ----------------
    dummy = types.SimpleNamespace(
        output_dir="web_output",
        cancel_processing=False,
        log_message=lambda m: print(m),
        progress_var=types.SimpleNamespace(set=lambda *a, **k: None),
        update_idletasks=lambda: None,
    )

    processor = FastProcessor(dummy, cfg, size_cfg)

    base = pathlib.Path(img_path).stem
    processor.generate(img_path, base)

    if not make_svg:
        svg_file = pathlib.Path(output_root) / base / f"{base}_template.svg"
        svg_file.unlink(missing_ok=True)

    return pathlib.Path(output_root) / base

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🎨 Fast Paint-by-Numbers Generator V5")
    print("=" * 40)
    
    try:
        if tk is None: 
            raise SystemExit("Tkinter not available in this environment.")
        app = FastPaintByNumbersApp()
        print("✅ Ready!")
        print("\nFeatures:")
        print("• Fast processing (minutes not hours)")
        print("• Clean single boundaries")
        print("• Tiny red numbers (4-10pt)")
        print("• Numbers fit in all regions")
        print("• No backgrounds or circles")
        print("• Professional output")
        
        app.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nInstall requirements:")
        print("pip install opencv-python pillow numpy scipy scikit-learn")

if __name__ == "__main__":
    main()
