import os
import shutil
import random
from pathlib import Path

# ================= CONFIGURATION =================
# Where your current folders are (e.g., dataset_training_1, dataset_training_2...)
SOURCE_ROOT = Path(r"C:\Users\moudi\Documents\Coding\Cat Camera\output")

# Where you want the FINAL combined dataset to go
DEST_ROOT = SOURCE_ROOT / "dataset_final"

# Split ratio (0.2 = 20% validation, 80% training)
VAL_RATIO = 0.2

# Random seed for reproducibility (so you get the same split every time if you re-run)
random.seed()

def main():
    # 1. Setup Destination Structure
    print(f"Creating destination structure at: {DEST_ROOT}")
    if DEST_ROOT.exists():
        print("WARNING: Destination folder already exists. Merging into it...")
    
    # Create YOLO standard folders
    for split in ['train', 'val']:
        (DEST_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (DEST_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 2. Find all Image-Label Pairs across all folders
    all_pairs = [] # List of tuples: (path_to_image, path_to_label)
    
    # Get list of dataset folders (dataset_training_1, _2, etc.)
    # We ignore 'dataset_final' to avoid recursion if you run it twice
    sub_folders = [f for f in SOURCE_ROOT.iterdir() if f.is_dir() and "dataset_training" in f.name]
    
    print(f"\nFound {len(sub_folders)} source folders: {[f.name for f in sub_folders]}")
    
    for folder in sub_folders:
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {folder.name} (Missing images or labels folder)")
            continue
            
        # Get all images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in image_files:
            # Construct expected label path
            label_path = labels_dir / (img_path.stem + ".txt")
            
            # Always add the pair - if label doesn't exist, we'll handle it during copy
            # (orphaned images are valid - they represent empty frames with no objects)
            all_pairs.append((img_path, label_path))

    total_files = len(all_pairs)
    print(f"\nTotal valid image/label pairs found: {total_files}")
    
    if total_files == 0:
        print("No files found! Check your paths.")
        return

    # 3. Shuffle and Split
    random.shuffle(all_pairs)
    
    val_count = int(total_files * VAL_RATIO)
    train_count = total_files - val_count
    
    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:]
    
    print(f"Splitting into:")
    print(f"  Training:   {len(train_pairs)} images")
    print(f"  Validation: {len(val_pairs)} images")
    
    # 4. Move Files
    def move_files(pairs, split_name):
        print(f"\nMoving {len(pairs)} files to {split_name}...")
        count = 0
        orphan_count = 0
        for img_src, lbl_src in pairs:
            # Define destination paths
            img_dest = DEST_ROOT / split_name / "images" / img_src.name
            lbl_dest = DEST_ROOT / split_name / "labels" / lbl_src.name
            
            # Handle duplicates (e.g. if dataset_1 and dataset_2 both have 'frame_0.jpg')
            if img_dest.exists():
                # Rename to make unique: frame_0_dup.jpg
                new_name = f"{img_src.stem}_{random.randint(1000,9999)}{img_src.suffix}"
                img_dest = DEST_ROOT / split_name / "images" / new_name
                lbl_dest = DEST_ROOT / split_name / "labels" / (Path(new_name).stem + ".txt")
            
            try:
                shutil.copy2(img_src, img_dest) # Use copy2 to keep metadata, or move() to cut-paste
                
                # If label doesn't exist, create an empty one (for images with no objects)
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dest)
                else:
                    # Create empty label file for orphaned images (no objects detected)
                    lbl_dest.touch()
                    orphan_count += 1
                    
                count += 1
            except Exception as e:
                print(f"Error moving {img_src.name}: {e}")
                
            if count % 100 == 0:
                print(f"  Moved {count}...")
        
        if orphan_count > 0:
            print(f"  Created {orphan_count} empty label files for images without objects")

    move_files(train_pairs, "train")
    move_files(val_pairs, "val")
    
    # 5. Create data.yaml automatically
    yaml_content = f"""path: {DEST_ROOT.absolute().as_posix()}
train: train/images
val: val/images

nc: 2
names: ['Simba', 'Nala']
"""
    
    with open(DEST_ROOT / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n" + "="*40)
    print("SUCCESS! Dataset is ready.")
    print(f"Location: {DEST_ROOT}")
    print(f"data.yaml created automatically.")
    print("="*40)

if __name__ == "__main__":
    main()