"""
Reference Image and Mask Generator for AI VDI System
This script allows capturing reference images and creating/editing masks for each inspection side.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import os
from datetime import datetime
import json


class ReferenceImageCapture:
    """Class for capturing reference images using camera or loading from file"""
    
    def __init__(self, num_frames=30, blur_kernel_size=5):
        self.num_frames = num_frames
        self.blur_kernel_size = blur_kernel_size
        self.reference_image = None
        
    def capture_from_camera(self, camera_index=1):
        """Capture reference image from camera"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return None
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"📹 Camera opened. Position object and press SPACE to capture {self.num_frames} frames")
        print("Press 'q' to quit, 's' to take single shot")
        
        capture_started = False
        captured_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
                
            # Display current frame with instructions
            display_frame = frame.copy()
            if not capture_started:
                cv2.putText(display_frame, "Press SPACE for multi-frame, S for single shot", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Q to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                progress = f"Captured: {len(captured_frames)}/{self.num_frames}"
                cv2.putText(display_frame, progress, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow("Reference Image Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not capture_started:
                capture_started = True
                print("🎯 Starting multi-frame capture...")
                
            elif key == ord('s'):
                # Single shot capture
                blurred_frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)
                self.reference_image = blurred_frame
                print("📸 Single frame captured")
                break
                
            elif key == ord('q'):
                break
                
            # Multi-frame capture
            if capture_started and len(captured_frames) < self.num_frames:
                blurred_frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)
                captured_frames.append(blurred_frame.astype(np.float32))
                
                if len(captured_frames) >= self.num_frames:
                    # Average all frames
                    self.reference_image = np.mean(captured_frames, axis=0).astype(np.uint8)
                    print("✅ Multi-frame capture complete")
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        return self.reference_image
    
    def load_from_file(self, filepath):
        """Load reference image from file"""
        self.reference_image = cv2.imread(filepath)
        if self.reference_image is None:
            print(f"❌ Error: Could not load image from {filepath}")
            return None
        print(f"✅ Image loaded from {filepath}")
        return self.reference_image


class MaskEditor:
    """Interactive mask editor with GUI"""
    
    def __init__(self, image):
        self.original_image = image.copy()
        self.image_height, self.image_width = image.shape[:2]
        
        # Create mask (white = inspect area, black = ignore area)
        self.mask = np.ones((self.image_height, self.image_width), dtype=np.uint8) * 255
        
        # Drawing parameters
        self.brush_size = 20
        self.drawing = False
        self.erase_mode = False  # False = paint white (inspect), True = paint black (ignore)
        self.last_x, self.last_y = 0, 0
        
        # Undo functionality
        self.undo_stack = []
        self.max_undo_steps = 20
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI for mask editing"""
        self.root = tk.Tk()
        self.root.title("Mask Editor - White=Inspect, Black=Ignore")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Brush controls
        brush_frame = ttk.LabelFrame(control_frame, text="Brush Controls")
        brush_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(brush_frame, from_=5, to=100, orient=tk.HORIZONTAL, 
                                    command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        
        # Mode controls
        mode_frame = ttk.LabelFrame(control_frame, text="Mode")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="inspect")
        ttk.Radiobutton(mode_frame, text="Inspect Area (White)", variable=self.mode_var, 
                       value="inspect", command=self.update_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Ignore Area (Black)", variable=self.mode_var, 
                       value="ignore", command=self.update_mode).pack(side=tk.LEFT)
        
        # Action buttons
        action_frame = ttk.LabelFrame(control_frame, text="Actions")
        action_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Clear All", command=self.clear_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Fill All", command=self.fill_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Auto Edge", command=self.auto_edge_mask).pack(side=tk.LEFT, padx=2)
        
        # Save/Load buttons
        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame, text="Save Mask", command=self.save_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Load Mask", command=self.load_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Preview", command=self.preview_masked_image).pack(side=tk.LEFT, padx=2)
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Bind keyboard events
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_mask())
        
        # Initial display
        self.update_display()
        
    def update_brush_size(self, value):
        """Update brush size"""
        self.brush_size = int(float(value))
        
    def update_mode(self):
        """Update drawing mode"""
        self.erase_mode = (self.mode_var.get() == "ignore")
        
    def save_undo_state(self):
        """Save current mask state for undo"""
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
            
    def undo(self):
        """Undo last operation"""
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.update_display()
            
    def clear_mask(self):
        """Clear mask (set all to ignore - black)"""
        self.save_undo_state()
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        self.update_display()
        
    def fill_mask(self):
        """Fill mask (set all to inspect - white)"""
        self.save_undo_state()
        self.mask = np.ones((self.image_height, self.image_width), dtype=np.uint8) * 255
        self.update_display()
        
    def auto_edge_mask(self):
        """Create automatic edge-based mask"""
        self.save_undo_state()
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to create thicker lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert (edges become inspection areas)
        self.mask = 255 - edges
        
        self.update_display()
        messagebox.showinfo("Auto Edge", "Automatic edge mask created. Edges are set to inspect areas.")
        
    def start_draw(self, event):
        """Start drawing"""
        self.drawing = True
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)
        self.save_undo_state()
        
    def draw(self, event):
        """Draw on mask"""
        if self.drawing:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Convert canvas coordinates to image coordinates
            img_x = int(x)
            img_y = int(y)
            
            if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
                # Draw circle on mask
                color = 0 if self.erase_mode else 255
                cv2.circle(self.mask, (img_x, img_y), self.brush_size // 2, color, -1)
                
                # Draw line from last position for smooth drawing
                if self.last_x is not None and self.last_y is not None:
                    cv2.line(self.mask, (int(self.last_x), int(self.last_y)), 
                            (img_x, img_y), color, self.brush_size)
                
                self.last_x, self.last_y = x, y
                self.update_display()
                
    def stop_draw(self, event):
        """Stop drawing"""
        self.drawing = False
        self.last_x = self.last_y = None
        
    def update_display(self):
        """Update the canvas display"""
        # Create overlay image
        overlay = self.original_image.copy()
        
        # Create colored mask overlay
        mask_colored = np.zeros_like(self.original_image)
        mask_colored[:, :, 1] = self.mask  # Green channel for inspect areas
        mask_colored[:, :, 2] = 255 - self.mask  # Red channel for ignore areas
        
        # Blend original image with mask
        alpha = 0.3
        overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
        
        # Convert to PIL Image for tkinter
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(overlay_rgb)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def save_mask(self):
        """Save mask to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            cv2.imwrite(filename, self.mask)
            messagebox.showinfo("Saved", f"Mask saved to {filename}")
            
    def load_mask(self):
        """Load mask from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            loaded_mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if loaded_mask is not None:
                self.save_undo_state()
                self.mask = cv2.resize(loaded_mask, (self.image_width, self.image_height))
                self.update_display()
                messagebox.showinfo("Loaded", f"Mask loaded from {filename}")
            else:
                messagebox.showerror("Error", "Could not load mask file")
                
    def preview_masked_image(self):
        """Preview the masked image"""
        # Apply mask to original image
        masked_image = self.original_image.copy()
        
        # Create 3-channel mask
        mask_3ch = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask (darken ignored areas)
        masked_image = cv2.bitwise_and(masked_image, mask_3ch)
        
        # Add ignored areas in red
        ignored_areas = 255 - self.mask
        red_overlay = np.zeros_like(masked_image)
        red_overlay[:, :, 2] = ignored_areas
        
        preview = cv2.addWeighted(masked_image, 0.8, red_overlay, 0.2, 0)
        
        # Show preview
        cv2.imshow("Mask Preview - Dark areas are ignored", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def run(self):
        """Run the mask editor"""
        self.root.mainloop()
        return self.mask


class ReferenceAndMaskGenerator:
    """Main class for generating reference images and masks"""
    
    def __init__(self):
        self.sides = ["Front", "Back", "Left", "Right"]
        self.current_side = 0
        self.project_dir = self.get_project_directory()
        self.reference_dir = os.path.join(self.project_dir, "data", "reference_images")
        self.mask_dir = os.path.join(self.project_dir, "data", "mask_images")
        
        # Create directories if they don't exist
        os.makedirs(self.reference_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        
        self.image_capture = ReferenceImageCapture()
        
    def get_project_directory(self):
        """Get the AI_VDI_System project directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming this script is in the project root
        return current_dir
        
    def show_instructions(self):
        """Show instructions for the user"""
        instructions = """
        🎯 Reference Image and Mask Generator
        ====================================
        
        This tool will help you create reference images and masks for each side of your product.
        
        Process:
        1. Position your product to show one side clearly
        2. Capture reference image (camera or load file)
        3. Create/edit mask to define inspection areas
        4. Repeat for all 6 sides
        
        Mask Editing:
        - White areas = INSPECT (will be analyzed)
        - Black areas = IGNORE (will be skipped)
        - Use brush to paint inspection areas
        - Use auto-edge detection for quick start
        
        Controls:
        - Mouse: Draw on mask
        - Ctrl+Z: Undo
        - Ctrl+S: Save mask
        
        Press Enter to continue...
        """
        print(instructions)
        input()
        
    def capture_side_reference(self, side_name):
        """Capture reference image for a specific side"""
        print(f"\n📸 Capturing reference image for {side_name} side")
        print("Choose capture method:")
        print("1. Camera capture")
        print("2. Load from file")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print(f"Position the product to show the {side_name} side clearly")
            input("Press Enter when ready...")
            image = self.image_capture.capture_from_camera()
        elif choice == "2":
            print("Select image file...")
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title=f"Select {side_name} reference image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
            )
            root.destroy()
            if filepath:
                image = self.image_capture.load_from_file(filepath)
            else:
                print("No file selected")
                return False
        else:
            print("Invalid choice")
            return False
            
        if image is None:
            print("Failed to capture/load image")
            return False
            
        # Save reference image
        ref_filename = os.path.join(self.reference_dir, f"{side_name.lower()}_reference.jpg")
        cv2.imwrite(ref_filename, image)
        print(f"✅ Reference image saved: {ref_filename}")
        
        # Create/edit mask
        print(f"\n🎨 Creating mask for {side_name} side")
        print("Opening mask editor...")
        
        mask_editor = MaskEditor(image)
        mask = mask_editor.run()
        
        if mask is not None:
            # Save mask
            mask_filename = os.path.join(self.mask_dir, f"{side_name.lower()}_mask.jpg")
            cv2.imwrite(mask_filename, mask)
            print(f"✅ Mask saved: {mask_filename}")
            return True
        else:
            print("❌ Mask creation cancelled")
            return False
            
    def generate_all_references(self):
        """Generate reference images and masks for all sides"""
        self.show_instructions()
        
        print("🚀 Starting reference generation for all sides...")
        
        for i, side in enumerate(self.sides):
            print(f"\n{'='*50}")
            print(f"Side {i+1}/{len(self.sides)}: {side}")
            print(f"{'='*50}")
            
            success = self.capture_side_reference(side)
            
            if success:
                print(f"✅ {side} side completed successfully")
            else:
                print(f"❌ {side} side failed")
                retry = input("Retry this side? (y/n): ").strip().lower()
                if retry == 'y':
                    success = self.capture_side_reference(side)
                    
            if not success:
                print(f"⚠️ Skipping {side} side")
                
        print("\n🎉 Reference generation process completed!")
        print(f"📁 Reference images saved in: {self.reference_dir}")
        print(f"📁 Mask images saved in: {self.mask_dir}")
        
    def generate_single_side(self):
        """Generate reference and mask for a single side"""
        print("Available sides:")
        for i, side in enumerate(self.sides):
            print(f"{i+1}. {side}")
            
        try:
            choice = int(input("Select side number: ")) - 1
            if 0 <= choice < len(self.sides):
                side = self.sides[choice]
                success = self.capture_side_reference(side)
                if success:
                    print(f"✅ {side} side completed successfully")
                else:
                    print(f"❌ {side} side failed")
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")
            
    def list_existing_files(self):
        """List existing reference images and masks"""
        print("\n📋 Existing Files:")
        print("-" * 40)
        
        for side in self.sides:
            side_lower = side.lower()
            ref_file = os.path.join(self.reference_dir, f"{side_lower}_reference.jpg")
            mask_file = os.path.join(self.mask_dir, f"{side_lower}_mask.jpg")
            
            ref_exists = "✅" if os.path.exists(ref_file) else "❌"
            mask_exists = "✅" if os.path.exists(mask_file) else "❌"
            
            print(f"{side:>8}: Reference {ref_exists} | Mask {mask_exists}")
            
    def main_menu(self):
        """Show main menu and handle user choice"""
        while True:
            print("\n" + "="*50)
            print("🎯 REFERENCE IMAGE & MASK GENERATOR")
            print("="*50)
            print("1. Generate all sides (complete workflow)")
            print("2. Generate single side")
            print("3. List existing files")
            print("4. Edit existing mask")
            print("5. Exit")
            print("-" * 50)
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                self.generate_all_references()
            elif choice == "2":
                self.generate_single_side()
            elif choice == "3":
                self.list_existing_files()
            elif choice == "4":
                self.edit_existing_mask()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
    def edit_existing_mask(self):
        """Edit an existing mask"""
        print("Select side to edit mask:")
        for i, side in enumerate(self.sides):
            print(f"{i+1}. {side}")
            
        try:
            choice = int(input("Select side number: ")) - 1
            if 0 <= choice < len(self.sides):
                side = self.sides[choice]
                side_lower = side.lower()
                
                ref_file = os.path.join(self.reference_dir, f"{side_lower}_reference.jpg")
                mask_file = os.path.join(self.mask_dir, f"{side_lower}_mask.jpg")
                
                if not os.path.exists(ref_file):
                    print(f"❌ Reference image not found for {side}")
                    return
                    
                # Load reference image
                image = cv2.imread(ref_file)
                mask_editor = MaskEditor(image)
                
                # Load existing mask if available
                if os.path.exists(mask_file):
                    existing_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if existing_mask is not None:
                        mask_editor.mask = cv2.resize(existing_mask, (mask_editor.image_width, mask_editor.image_height))
                        mask_editor.update_display()
                        
                # Edit mask
                mask = mask_editor.run()
                
                if mask is not None:
                    cv2.imwrite(mask_file, mask)
                    print(f"✅ Mask updated: {mask_file}")
                else:
                    print("❌ Mask editing cancelled")
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")


def main():
    """Main function"""
    print("🎯 AI VDI System - Reference Image & Mask Generator")
    print("=" * 55)
    
    generator = ReferenceAndMaskGenerator()
    generator.main_menu()


if __name__ == "__main__":
    main()