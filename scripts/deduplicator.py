import cv2
import numpy as np
import os
from pathlib import Path

# ================= CONFIGURATION =================
dataset_path = Path(r"C:\Users\moudi\Documents\Coding\Cat Camera\output\dataset_training_7\images")

# --- MODE SWITCH ---
# Set to True to SEE the differences window.
# Set to False to SCAN and DELETE files.
DEBUG_MODE = True  

# TUNING
SIMILARITY_THRESHOLD = 15.0  # < 15.0 means "Duplicate" (Lower = Stricter)
SKIP_FIRST_N = 50           # Ignore the first 50 labeled images

# ================= HELPERS =================
def get_image_files(folder):
    return sorted(list(folder.glob("*.jpg")))

def select_tv_region(image_path):
    """Let user draw the TV box."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load {image_path}")
    
    print("Opening TV selection window...")
    cv2.namedWindow("1. Draw Box on TV (Space to Confirm)", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("1. Draw Box on TV (Space to Confirm)", img, showCrosshair=True)
    cv2.destroyAllWindows()
    return r

def mse(imageA, imageB):
    """Calculate pixel difference score (Lower = More Similar)."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# ================= DEBUG MODE LOGIC =================
def run_debug_mode(image_files, x, y, w, h):
    print("\n--- DEBUG MODE ACTIVE (Stacked Layout) ---")
    print(" * Visualizing first 20 changes.")
    print(" * Press SPACE to next image.")
    print(" * Press 'q' to quit.")
    print(f" * Current Threshold: {SIMILARITY_THRESHOLD}\n")

    mask_tv = (w > 0 and h > 0)

    # Load baseline
    prev_img = cv2.imread(str(image_files[SKIP_FIRST_N]))
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    if mask_tv: prev_gray[y:y+h, x:x+w] = 0

    # Loop through next 20 images only
    limit = min(SKIP_FIRST_N + 21, len(image_files))
    
    for i in range(SKIP_FIRST_N + 1, limit):
        curr_file = image_files[i]
        curr_img = cv2.imread(str(curr_file))
        if curr_img is None: continue
        
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        if mask_tv: 
            curr_gray[y:y+h, x:x+w] = 0
            # Draw box on visual output only
            cv2.rectangle(curr_img, (x, y), (x+w, y+h), (0, 0, 0), 5)

        score = mse(prev_gray, curr_gray)
        is_dup = score < SIMILARITY_THRESHOLD

        # Visualize Logic
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_color = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2BGR)

        # Labels
        color = (0, 0, 255) if is_dup else (0, 255, 0) # Red=Delete, Green=Keep
        status = "DELETE" if is_dup else "KEEP"
        
        # --- BIG TEXT BLOCK (On Current Image) ---
        # Status (DELETE/KEEP)
        cv2.putText(curr_img, status, (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 10)
        # Score
        cv2.putText(curr_img, f"Score: {score:.2f}", (50, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        # Limit
        cv2.putText(curr_img, f"Limit: {SIMILARITY_THRESHOLD}", (50, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

        # --- NEW STACKED LAYOUT ---
        # Define height for the smaller rows on the left (reduced to fit screen)
        row_height = 250

        # Calculate dimensions for small images (Prev and Diff)
        ratio_small = row_height / prev_img.shape[0]
        dim_small = (int(prev_img.shape[1] * ratio_small), row_height)

        # Resize Prev and Diff
        prev_small = cv2.resize(prev_img, dim_small)
        diff_small = cv2.resize(diff_color, dim_small)

        # Stack Left Column (Previous on top of Diff)
        left_col = np.vstack((prev_small, diff_small))

        # Calculate dimensions for the Current image on the right
        # Match the exact height of the left column
        total_height = left_col.shape[0]
        ratio_large = total_height / curr_img.shape[0]
        dim_large = (int(curr_img.shape[1] * ratio_large), total_height)

        # Resize Current to match left column height exactly
        curr_large = cv2.resize(curr_img, dim_large)

        # Combine horizontally
        combined = np.hstack((left_col, curr_large))
        
        cv2.imshow("Debug: Left(Prev/Diff) | Right(Current)", combined)
        
        # In Debug, we update 'prev' only if it's NOT a duplicate
        if not is_dup:
            prev_gray = curr_gray
            prev_img = curr_img
        
        key = cv2.waitKey(0)
        if key == ord('q'): break

    cv2.destroyAllWindows()
    print("Debug finished. If results look good, set DEBUG_MODE = False.")

# ================= BATCH MODE LOGIC =================
def run_batch_mode(image_files, x, y, w, h):
    print("\n--- BATCH MODE ACTIVE ---")
    mask_tv = (w > 0 and h > 0)
    files_to_delete = []

    # Initialize
    prev_img = cv2.imread(str(image_files[SKIP_FIRST_N]))
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    if mask_tv: prev_gray[y:y+h, x:x+w] = 0

    total_files = len(image_files)
    
    for i in range(SKIP_FIRST_N + 1, total_files):
        curr_file = image_files[i]
        curr_img = cv2.imread(str(curr_file))
        if curr_img is None: continue
        
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        if mask_tv: curr_gray[y:y+h, x:x+w] = 0

        score = mse(prev_gray, curr_gray)

        if score < SIMILARITY_THRESHOLD:
            # It's a duplicate
            files_to_delete.append(curr_file)
        else:
            # Significant change, update baseline
            prev_gray = curr_gray

        if i % 100 == 0:
            print(f"Processed {i}/{total_files}...")

    # Confirmation
    count = len(files_to_delete)
    print(f"\nFound {count} duplicates out of {total_files - SKIP_FIRST_N} images.")
    
    if count > 0:
        ans = input("PERMANENTLY DELETE these files? (yes/no): ").lower()
        if ans == "yes":
            for f in files_to_delete:
                try:
                    f.unlink() # Delete Image
                    lbl = f.parent.parent / "labels" / (f.stem + ".txt")
                    if lbl.exists(): lbl.unlink() # Delete Label
                except Exception as e:
                    print(e)
            print("Deletion complete.")
        else:
            print("Cancelled.")
    else:
        print("No duplicates found.")

# ================= MAIN =================
if __name__ == "__main__":
    files = get_image_files(dataset_path)
    if len(files) > SKIP_FIRST_N:
        # 1. Select TV (Happens in both modes)
        ref_idx = min(SKIP_FIRST_N, len(files)-1)
        rect = select_tv_region(files[ref_idx])
        
        # 2. Run Mode
        if DEBUG_MODE:
            run_debug_mode(files, *rect)
        else:
            run_batch_mode(files, *rect)