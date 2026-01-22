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
MOTION_THRESHOLD_PERCENTAGE = 0.0015  # Captures top 5% most active frames

# Cooldown: If we find a target, how many seconds to skip?
COOLDOWN_SECONDS = 1.5  # Balanced to avoid duplicates while maintaining diversity 

# Resize for motion check (Speed optimization - doesn't affect final image)
MOTION_RESOLUTION = (640, 360) 

# FILTER: Ignore boxes smaller than this % of the screen (removes tiny specks)
MIN_BOX_AREA_PERCENT = 0.0 

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# ================= MODEL LOADING =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"
REVISION = "21a599d414c4d928c9032694c424fb94458e3594"

print(f"Loading Teacher Model ({DEVICE})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, attn_implementation="eager", revision=REVISION
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)

TARGETS = [
    {"prompt": "orange cat", "id": 0, "name": "Simba"},
    {"prompt": "black cat",  "id": 1, "name": "Nala"}
]

def run_florence_inference(image_pil, text_prompt):
    """
    Run grounding inference to find bounding boxes for the given prompt.
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

def verify_crop(image_pil, bbox, target_name):
    """
    Crops the detected area and asks the model to describe it.
    Returns True if the description confirms it's a cat.
    """
    # 1. Crop the image based on the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    
    # Safety check for image boundaries
    w, h = image_pil.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # If crop is invalid (too small/inverted), reject
    if x2 - x1 < 10 or y2 - y1 < 10: 
        return False, "invalid_crop"

    crop = image_pil.crop((x1, y1, x2, y2))
    
    # 2. Run Simple Captioning on the crop
    task_prompt = "<CAPTION>"
    results = processor(text=task_prompt, images=crop, return_tensors="pt")
    
    input_ids = results["input_ids"].to(DEVICE)
    pixel_values = results["pixel_values"].to(DEVICE)

    generated_ids = model.generate(
        input_ids=input_ids, pixel_values=pixel_values,
        max_new_tokens=256, use_cache=False, num_beams=1
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(crop.width, crop.height)
    )
    
    caption = parsed_answer[task_prompt].lower()
    
    # 3. Validation Logic
    # We check for generic terms because a close-up might just be "a cat face"
    valid_keywords = ["cat", "kitten", "feline", "animal", "mammal", "pet", target_name.split()[0].lower()]
    
    # Reject common false positives explicitly
    invalid_keywords = ["pillow", "cushion", "shadow", "floor", "blanket", "shoe"]
    
    is_valid = any(word in caption for word in valid_keywords)
    # Optional: stricter check (if it contains an invalid word, reject it)
    # if any(word in caption for word in invalid_keywords): is_valid = False
    
    return is_valid, caption

def get_motion_score(current_frame, prev_frame_small):
    curr_small = cv2.resize(current_frame, MOTION_RESOLUTION)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    if prev_frame_small is None:
        return 0.0, curr_gray

    delta = cv2.absdiff(prev_frame_small, curr_gray)
    
    # Calculate percentage of screen changed
    # total_pixels = thresh.size
    # changed_pixels = np.count_nonzero(thresh)
    # score = (changed_pixels / total_pixels) * 100

    # Threshold: highlight changed pixels (value 25 is arbitrary threshold for pixel intensity)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
    score = (np.count_nonzero(thresh) / thresh.size) * 100
    
    return score, curr_gray

def process_videos():
    video_files = sorted(list(Path(INPUT_VIDEO_FOLDER).glob("*.mp4")))
    total_samples = 0
    
    print(f"Starting Smart Motion Auto-Labeling on {len(video_files)} videos...")
    print(f"Input folder: {INPUT_VIDEO_FOLDER}")
    print(f"Output folder: {DATASET_DIR}")
    print(f"Motion threshold: {MOTION_THRESHOLD_PERCENTAGE}%")
    print(f"Cooldown: {COOLDOWN_SECONDS}s")
    print(f"Method: Direct Grounding -> Crop Verification")

    for video_idx, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if video_idx != 6:
        #     continue  # TEMP SKIP FOR TESTING

        # Calculate frames to jump for cooldown
        cooldown_frames = int(fps * COOLDOWN_SECONDS)
        
        print(f"\n[Video {video_idx}/{len(video_files)}] {video_path.name} | FPS: {fps} | Total Frames: {total_frames}")
        print(f"  Cooldown frames: {cooldown_frames}")
        
        prev_gray_small = None
        frame_idx = 0
        frames_checked = 0
        frames_with_motion = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- 1. FAST MOTION CHECK ---
            motion_score, current_gray_small = get_motion_score(frame, prev_gray_small)
            prev_gray_small = current_gray_small 
            frames_checked += 1
            
            # Progress update every 100 frames
            if frames_checked % 100 == 0:
                print(f"  > Frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%) | Motion: {motion_score:.6f}% | Detected motion in {frames_with_motion} frames so far")
            
            if motion_score < MOTION_THRESHOLD_PERCENTAGE:
                frame_idx += 1
                continue
            
            frames_with_motion += 1
            print(f"  >> MOTION DETECTED at frame {frame_idx} ({frame_idx/fps:.1f}s) | Score: {motion_score:.6f}%")
                
            # --- 2. HEAVY LIFTING (Only runs if motion detected) ---
            print(f"  >> Running AI detection...")
            
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = image_pil.size
            yolo_labels = []
            has_detection = False
            
            # Check for each target directly (skipping the "is it there?" check)
            for target in TARGETS:
                prediction = run_florence_inference(image_pil, target["prompt"])
                bboxes = prediction.get('bboxes', [])
                print(f"Found {len(bboxes)} bbox(es)")
                
                for bbox_idx, bbox in enumerate(bboxes, 1):
                    x1, y1, x2, y2 = bbox
                    
                    # A. Filter by Size (Ignore tiny specks/noise)
                    box_area_percent = ((x2 - x1) * (y2 - y1)) / (w * h)
                    if box_area_percent < MIN_BOX_AREA_PERCENT:
                        continue 
                    
                    # B. Verify Content (Crop & Check)
                    is_valid, crop_caption = verify_crop(image_pil, bbox, target["prompt"])
                    
                    if is_valid:
                        # Normalize for YOLO
                        b_cx = ((x1 + x2) / 2) / w
                        b_cy = ((y1 + y2) / 2) / h
                        b_w = (x2 - x1) / w
                        b_h = (y2 - y1) / h
                        
                        yolo_labels.append(f"{target['id']} {b_cx:.6f} {b_cy:.6f} {b_w:.6f} {b_h:.6f}")
                        has_detection = True
                        print(f"       ✓ FOUND {target['name']} (Verified: '{crop_caption}')")
                    else:
                        # print(f"       ✗ Rejected false positive (Caption: '{crop_caption}')")
                        pass

            # --- 3. SAVE & COOLDOWN ---
            if has_detection:
                base_name = f"{video_path.stem}_{frame_idx}"
                image_path = os.path.join(IMAGES_DIR, f"{base_name}.jpg")
                label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
                
                cv2.imwrite(image_path, frame)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_labels))
                
                total_samples += 1
                print(f"  [+] ✓ SAVED Sample #{total_samples} at {frame_idx/fps:.1f}s (Motion: {motion_score:.4f}%)")
                print(f"      Image: {image_path}")
                print(f"      Label: {label_path} ({len(yolo_labels)} detection(s))")
                
                # Jump forward
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = current_pos + cooldown_frames
                if new_pos < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_idx = int(new_pos)
                    print(f"      Jumping ahead {cooldown_frames} frames to frame {frame_idx}")
                    # Reset motion baseline because we jumped (scene might have changed)
                    prev_gray_small = None 
                else:
                    print(f"      Reached end of video after cooldown jump")
                    break # End of video
            else:
                print(f"     ✗ No detections found despite motion - continuing...")
                frame_idx += 1

        cap.release()
        print(f"  Video complete: {frames_checked} frames checked, {frames_with_motion} had motion above threshold")

    print(f"\n" + "="*60)
    print(f"DISTILLATION COMPLETE")
    print(f"="*60)
    print(f"Total samples generated: {total_samples}")
    print(f"Output directory: {DATASET_DIR}")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Labels saved to: {LABELS_DIR}")
    
    # Verify files were created
    import glob
    image_count = len(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    label_count = len(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    print(f"\nVerification:")
    print(f"  Images on disk: {image_count}")
    print(f"  Labels on disk: {label_count}")

# ================= 5. MAIN EXECUTION =================
if __name__ == "__main__":
    # We wrap the main call in the silencer
    print("Initializing... (FFmpeg warnings will be suppressed)")
    with SuppressStderr():
        process_videos()
