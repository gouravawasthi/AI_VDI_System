# Reference Image & Mask Generator - Usage Guide

## Overview
This script helps you create reference images and inspection masks for the AI VDI System. You'll capture or load reference images for each side of your product and create masks to define which areas should be inspected.

## Features

### 📸 **Image Capture**
- **Camera Capture**: Live camera feed with multi-frame averaging for noise reduction
- **Single Shot**: Quick single frame capture
- **File Loading**: Load existing images from files

### 🎨 **Mask Editor**
- **Interactive GUI**: Visual mask editing with real-time preview
- **Brush Tools**: Adjustable brush size for painting inspection areas
- **Two Modes**:
  - **Inspect Areas (White)**: Areas that will be analyzed during inspection
  - **Ignore Areas (Black)**: Areas that will be skipped during inspection
- **Auto Edge Detection**: Automatic mask generation based on image edges
- **Undo/Redo**: Full undo functionality for mistake correction

### 💾 **File Management**
- Automatic saving to correct directories
- Load/edit existing masks
- Preview functionality to see mask effects

## How to Use

### 1. Run the Script
```bash
cd /Users/gourav/Desktop/Taisys/AI_VDI_System
python reference_generator.py
```

### 2. Main Menu Options

#### **Option 1: Generate All Sides**
- Complete workflow for all 6 sides (Front, Back, Left, Right, Top, Bottom)
- For each side:
  1. Position your product to show that side clearly
  2. Choose capture method (camera or file)
  3. Capture/load the reference image
  4. Edit the mask to define inspection areas
  5. Save and move to next side

#### **Option 2: Generate Single Side**
- Work on one specific side only
- Same process as above but for selected side

#### **Option 3: List Existing Files**
- Shows which reference images and masks already exist
- Helps track progress

#### **Option 4: Edit Existing Mask**
- Modify masks for existing reference images
- Useful for fine-tuning inspection areas

### 3. Camera Capture Process

1. **Position Product**: Place product showing the desired side
2. **Camera View**: Live camera feed will open
3. **Capture Options**:
   - **SPACE**: Start multi-frame capture (30 frames averaged for noise reduction)
   - **S**: Single shot capture
   - **Q**: Quit without capturing

### 4. Mask Editing Process

1. **Mask Editor Opens**: Shows reference image with overlay
2. **Color Coding**:
   - **Green Overlay**: Inspection areas (white in mask)
   - **Red Overlay**: Ignored areas (black in mask)
3. **Tools Available**:
   - **Brush Size Slider**: Adjust brush size (5-100 pixels)
   - **Mode Selection**: 
     - "Inspect Area" - Paint white (analyze these areas)
     - "Ignore Area" - Paint black (skip these areas)
   - **Actions**:
     - **Clear All**: Set entire mask to ignore (black)
     - **Fill All**: Set entire mask to inspect (white)
     - **Undo**: Undo last operation
     - **Auto Edge**: Automatic edge-based mask generation
   - **File Operations**:
     - **Save Mask**: Save current mask
     - **Load Mask**: Load existing mask
     - **Preview**: See how mask affects the image

### 5. Keyboard Shortcuts

- **Mouse**: Draw on mask
- **Ctrl+Z**: Undo last operation
- **Ctrl+S**: Save mask
- **Close Window**: Finish editing and save

## File Organization

The script automatically organizes files:

```
AI_VDI_System/
├── data/
│   ├── reference_images/
│   │   ├── front_reference.jpg
│   │   ├── back_reference.jpg
│   │   ├── left_reference.jpg
│   │   ├── right_reference.jpg
│   │   ├── top_reference.jpg
│   │   └── bottom_reference.jpg
│   └── mask_images/
│       ├── front_mask.jpg
│       ├── back_mask.jpg
│       ├── left_mask.jpg
│       ├── right_mask.jpg
│       ├── top_mask.jpg
│       └── bottom_mask.jpg
└── reference_generator.py
```

## Tips for Best Results

### **Reference Images**
1. **Good Lighting**: Ensure even, consistent lighting
2. **Clear Focus**: Product should be in sharp focus
3. **Stable Position**: Minimize camera shake
4. **Consistent Distance**: Maintain same distance for all sides
5. **Clean Background**: Simple, uncluttered background

### **Mask Creation**
1. **Start with Auto Edge**: Use automatic edge detection as starting point
2. **Focus on Important Areas**: Mark critical inspection zones as inspect areas
3. **Ignore Backgrounds**: Mark backgrounds and irrelevant areas as ignore
4. **Consider Tolerances**: Account for slight position variations
5. **Test and Refine**: Use preview to check mask effectiveness

### **Workflow Tips**
1. **Complete One Side at a Time**: Finish both reference and mask before moving on
2. **Consistent Setup**: Use same camera position and lighting for all sides
3. **Save Frequently**: The system auto-saves, but you can manually save masks
4. **Review Results**: Use list option to check progress

## Troubleshooting

### **Camera Issues**
- Ensure camera is connected and working
- Try different camera index if multiple cameras available
- Check camera permissions

### **Image Quality**
- Adjust lighting for better contrast
- Use multi-frame capture for noise reduction
- Ensure product is properly focused

### **Mask Editor**
- If mask editor doesn't respond, check if window has focus
- Use preview to verify mask is working correctly
- Remember: white = inspect, black = ignore

### **File Issues**
- Check directory permissions
- Ensure sufficient disk space
- Verify file paths are correct

## Integration with AI VDI System

Once you've created all reference images and masks:

1. **Automatic Loading**: The AI VDI System will automatically load these files
2. **Inspection Process**: During inspection, the system will:
   - Compare live camera feed with reference images
   - Apply masks to focus on relevant areas
   - Highlight differences using gradient analysis
3. **Real-time Analysis**: Live comparison shows original feed + processed overlay

## Next Steps

After generating references and masks:
1. Test the inspection system with your reference images
2. Fine-tune masks based on inspection results
3. Adjust camera settings in the main system if needed
4. Document your setup for consistency

---

**Note**: This tool is designed to work seamlessly with the AI VDI System's advanced inspection window, which uses the generated reference images and masks for real-time quality inspection.