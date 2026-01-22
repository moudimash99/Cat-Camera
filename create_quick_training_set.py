import cv2
import torch
import os
import sys
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# ================= 1. SILENCER CLASS =================
class SuppressStderr:
    """
    Redirects low-level C libraries (like FFmpeg) to the void
    so they don't spam the console.
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(2)] # Save the original stderr (2)

    def __enter__(self):
        os.dup2(self.null_fds[1], 2) # Redirect stderr to null

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2) # Restore stderr
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# ================= 2. CONFIGURATION =================
INPUT_VIDEO_FOLDER = "./downloads_mini"
OUTPUT_DIR = "./output"
BASE_DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset_training")

# Find the next available incremental folder ID
def get_next_dataset_folder():
    """Find the next available incremental folder ID"""
    max_id = -1
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for any existing numbered folders
    if os.path.exists(OUTPUT_DIR):
        for item in os.listdir(OUTPUT_DIR):
            if item.startswith("dataset_training_") and os.path.isdir(os.path.join(OUTPUT_DIR, item)):
                try:
                    folder_id = int(item.split("_")[-1])
                    max_id = max(max_id, folder_id)
                except ValueError:
                    pass
    
    return max_id + 1

DATASET_ID = get_next_dataset_folder()
DATASET_DIR = os.path.join(OUTPUT_DIR, f"dataset_training_{DATASET_ID}")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Sensitivity: How much pixel change triggers the AI? (Lower = more sensitive)
# 1.0 means 1% of the screen changed pixels.
# Optimized for static security camera footage based on dataset analysis
MOTION_THRESHOLD_PERCENTAGE = 0.015  # Captures top 5% most active frames

# Cooldown: If we find a target, how many seconds to skip?
COOLDOWN_SECONDS = 1.5  # Balanced to avoid duplicates while maintaining diversity 

# Resize for motion check (Speed optimization - doesn't affect final image)
MOTION_RESOLUTION = (640, 360) 

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# ================= MODEL LOADING =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"
REVISION = "21a599d414c4d928c9032694c424fb94458e3594"

# Keywords to detect cats in captions (including plurals and common variations)
CAT_KEYWORDS = ["cat", "cats", "kitten", "kittens", "feline", "felines", "kitty", "kitties"]

print(f"Loading Teacher Model ({DEVICE})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, attn_implementation="eager", revision=REVISION
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)

TARGETS = [
    {"prompt": "orange cat", "id": 0, "name": "Simba"},
    {"prompt": "black cat",  "id": 1, "name": "Nala"}
]

def check_target_presence(image_pil):
    """
    Check if a cat is present in the image.
    Uses caption task to determine presence before running expensive grounding.
    Returns: True if a cat is likely present, False otherwise, and the caption
    """
    task_prompt = "<CAPTION>"
    results = processor(text=task_prompt, images=image_pil, return_tensors="pt")
    
    input_ids = results["input_ids"].to(DEVICE)
    pixel_values = results["pixel_values"].to(DEVICE)

    generated_ids = model.generate(
        input_ids=input_ids, pixel_values=pixel_values,
        max_new_tokens=256, use_cache=False, num_beams=1
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
    )
    
    caption = parsed_answer[task_prompt].lower()
    # Check if a cat is present in the image
    # Use lenient matching: just check if any cat-related keyword is mentioned
    # The subsequent grounding task will handle specific target matching (orange/black)
    has_cat = any(word in caption for word in CAT_KEYWORDS)
    
    return has_cat, caption

def run_florence_inference(image_pil, text_prompt):
    """
    Run grounding inference to find bounding boxes for the given prompt.
    Should only be called after confirming target presence.
    """
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    full_text = f"{task_prompt}{text_prompt}"
    results = processor(text=full_text, images=image_pil, return_tensors="pt")
    
    input_ids = results["input_ids"].to(DEVICE)
    pixel_values = results["pixel_values"].to(DEVICE)

    generated_ids = model.generate(
        input_ids=input_ids, pixel_values=pixel_values,
        max_new_tokens=1024, use_cache=False, num_beams=1
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
    )
    return parsed_answer[task_prompt]

def get_motion_score(current_frame, prev_frame_small):
    """
    Calculates percentage of changed pixels. 
    Returns: score (float), current_frame_small (for next iteration)
    """
    # Resize and blur to remove noise (wind/leaves)
    curr_small = cv2.resize(current_frame, MOTION_RESOLUTION)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    if prev_frame_small is None:
        return 0.0, curr_gray

    # Compute absolute difference
    delta = cv2.absdiff(prev_frame_small, curr_gray)
    
    # Threshold: highlight changed pixels (value 25 is arbitrary threshold for pixel intensity)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of screen changed
    total_pixels = thresh.size
    changed_pixels = np.count_nonzero(thresh)
    score = (changed_pixels / total_pixels) * 100
    
    return score, curr_gray

def process_videos():
    video_files = sorted(list(Path(INPUT_VIDEO_FOLDER).glob("*.mp4")))
    total_samples = 0
    
    print(f"Starting Smart Motion Auto-Labeling on {len(video_files)} videos...")

    for video_idx, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to jump for cooldown
        cooldown_frames = int(fps * COOLDOWN_SECONDS)
        
        print(f"\n[Video {video_idx}/{len(video_files)}] {video_path.name} | FPS: {fps}")
        
        prev_gray_small = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- 1. FAST MOTION CHECK ---
            motion_score, current_gray_small = get_motion_score(frame, prev_gray_small)
            
            # Update previous frame for next loop
            prev_gray_small = current_gray_small 
            
            # If motion is too low, skip this frame
            if motion_score < MOTION_THRESHOLD_PERCENTAGE:
                frame_idx += 1
                continue
                
            # --- 2. HEAVY LIFTING (Only runs if motion detected) ---
            # print(f"  > Motion detected ({motion_score:.2f}%) at {frame_idx/fps:.1f}s. Checking AI...")
            
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = image_pil.size
            yolo_labels = []
            has_detection = False
            
            # First, check if any cat is present in the image
            is_present, caption = check_target_presence(image_pil)
            
            if not is_present:
                # No cat detected, skip this frame
                frame_idx += 1
                continue
            
            # Cat detected! Now check which specific targets (orange/black) are present
            for target in TARGETS:
                prediction = run_florence_inference(image_pil, target["prompt"])
                bboxes = prediction.get('bboxes', [])
                
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    
                    # Normalize for YOLO
                    b_cx = ((x1 + x2) / 2) / w
                    b_cy = ((y1 + y2) / 2) / h
                    b_w = (x2 - x1) / w
                    b_h = (y2 - y1) / h
                    
                    yolo_labels.append(f"{target['id']} {b_cx:.6f} {b_cy:.6f} {b_w:.6f} {b_h:.6f}")
                    has_detection = True

            # --- 3. SAVE & COOLDOWN ---
            if has_detection:
                base_name = f"{video_path.stem}_{frame_idx}"
                cv2.imwrite(os.path.join(IMAGES_DIR, f"{base_name}.jpg"), frame)
                with open(os.path.join(LABELS_DIR, f"{base_name}.txt"), "w") as f:
                    f.write("\n".join(yolo_labels))
                
                total_samples += 1
                print(f"  [+] Captured Sample #{total_samples} at {frame_idx/fps:.1f}s (Motion: {motion_score:.1f}%)")
                
                # --- APPLY COOLDOWN CURSOR ---
                # Jump forward in the video file
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = current_pos + cooldown_frames
                
                if new_pos < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_idx = int(new_pos)
                    # Reset motion baseline because we jumped (scene might have changed)
                    prev_gray_small = None 
                else:
                    break # End of video
            else:
                frame_idx += 1

        cap.release()

    print(f"------------------------------------------------")
    print(f"Distillation Complete. {total_samples} labelled images generated.")
    print(f"Output saved to: {DATASET_DIR}")

# ================= 5. MAIN EXECUTION =================
if __name__ == "__main__":
    # We wrap the main call in the silencer
    print("Initializing... (FFmpeg warnings will be suppressed)")
    with SuppressStderr():
        process_videos()
