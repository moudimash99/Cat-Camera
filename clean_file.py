import os
import shutil
from pathlib import Path

def clean_empty_parent_folders(root_path):
    """Remove parent folders if they contain only 2 empty subdirectories"""
    root = Path(root_path)
    
    if not root.exists():
        print(f"Path does not exist: {root_path}")
        return
    
    for parent in root.iterdir():
        if not parent.is_dir():
            continue
        
        subdirs = [d for d in parent.iterdir() if d.is_dir()]
        
        # Check if exactly 2 subdirectories and both are empty
        if len(subdirs) == 2 and all(not any(d.iterdir()) for d in subdirs):
            try:
                shutil.rmtree(parent)
                print(f"Removed: {parent}")
            except Exception as e:
                print(f"Error removing {parent}: {e}")

# Usage
clean_empty_parent_folders(r"C:\Users\Theotime\Documents\GitHub\Cat-Camera\output")