import cv2
import numpy as np
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm 

# ================= CONFIGURATION =================
BASE_INPUT_DIR = "./output"
BASE_OUTPUT_DIR = "./cleaned_output"

# Class Definitions
CLASS_MAP = {
    0: "Simba", # Orange
    1: "Nala"   # Black
}

# ================= COLOR CHECK LOGIC =================
def verify_color_opencv(image_bgr, bbox, class_id):
    """
    Returns True if the box contains the correct color pixels.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Boundary Check
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 - x1 < 5 or y2 - y1 < 5: return False # Too small
    
    # Crop & Convert to HSV
    crop = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = crop.shape[0] * crop.shape[1]

    if class_id == 0: # SIMBA (ORANGE)
        # Looking for Orange/Yellow with HIGH Saturation 
        # (High saturation filters out the dull wooden floor)
        
        # Range 1: Red-Orange
        lower1 = np.array([0, 95, 80])
        upper1 = np.array([25, 255, 255])
        
        # Range 2: Red-Purple side (wrap around)
        lower2 = np.array([160, 95, 80])
        upper2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        threshold = 0.10 # 10% of box must be orange

    elif class_id == 1: # NALA (BLACK)
        # Looking for Dark pixels (Value < 60)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 60]) 
        
        mask = cv2.inRange(hsv, lower, upper)
        threshold = 0.15 # 15% of box must be black

    else:
        return False

    # Calculate Score
    match_pixels = cv2.countNonZero(mask)
    score = match_pixels / total_pixels
    
    return score > threshold

# ================= MAIN PROCESSING =================
def main():
    # 1. Get Dataset ID from user
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        dataset_id = input("Enter the dataset ID number to clean (e.g., 3): ").strip()

    # 2. Setup Paths
    input_dir = os.path.join(BASE_INPUT_DIR, f"dataset_training_{dataset_id}")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"dataset_training_{dataset_id}")

    images_in = os.path.join(input_dir, "images")
    labels_in = os.path.join(input_dir, "labels")
    
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")

    # 3. Validation
    if not os.path.exists(input_dir):
        print(f"Error: Input folder not found: {input_dir}")
        return

    # 4. Prepare Output Folders
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(images_out)
    os.makedirs(labels_out)

    # 5. Process
    label_files = sorted(list(Path(labels_in).glob("*.txt")))
    print(f"Processing ID {dataset_id}...")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(label_files)} labels to check.")

    kept_count = 0
    rejected_count = 0

    for label_file in tqdm(label_files):
        # Load Image
        image_name = label_file.stem + ".jpg" # Assuming jpg
        image_path = os.path.join(images_in, image_name)
        
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None: continue
        
        h, w = image.shape[:2]

        # Read Labels
        with open(label_file, "r") as f:
            lines = f.readlines()

        valid_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # Convert YOLO -> BBox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Check Color
            if verify_color_opencv(image, (x1, y1, x2, y2), class_id):
                valid_lines.append(line.strip())

        # Save ONLY if valid lines exist
        if valid_lines:
            shutil.copy(image_path, os.path.join(images_out, image_name))
            with open(os.path.join(labels_out, label_file.name), "w") as f:
                f.write("\n".join(valid_lines))
            kept_count += 1
        else:
            rejected_count += 1

    print("-" * 40)
    print(f"Cleaning Complete.")
    print(f"Kept Images: {kept_count}")
    print(f"Rejected Images: {rejected_count}")
    print(f"Cleaned dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()