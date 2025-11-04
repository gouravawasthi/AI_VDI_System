import cv2

def roi_selector(image_path):
    """
    Interactive ROI selector using OpenCV GUI.
    Allows drawing multiple ROIs and prints coordinates in (x, y, w, h) format.
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image.")
        return

    print("\n🖱️ INSTRUCTIONS:")
    print(" - Drag to select multiple ROIs.")
    print(" - Press ENTER or SPACE after each ROI.")
    print(" - Press ESC when done.\n")

    # Let user select multiple ROIs
    rois = cv2.selectROIs("Select ROIs", image, showCrosshair=True)
    cv2.destroyAllWindows()

    if len(rois) == 0:
        print("⚠️ No ROI selected.")
        return

    # Generate dictionary in pipeline format
    roi_dict = {}
    for i, roi in enumerate(rois, 1):
        x, y, w, h = roi
        roi_dict[f"roi{i}"] = (int(x), int(y), int(w), int(h))

    print("\n✅ ROI Definitions for your code:\n")
    print("roi_definitions = {")
    for k, v in roi_dict.items():
        print(f"    '{k}': {v},")
    print("}\n")

    return roi_dict


if __name__ == "__main__":
    image_path = input("Enter path to your image: ").strip()
    roi_definitions = roi_selector(image_path)
