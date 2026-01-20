import cv2
import torch
import os
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# ================= CONFIGURATION =================
INPUT_VIDEO_FOLDER = "./downloads"
DATASET_DIR = "./dataset_training"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# -------------------------------------------------
# TEACHER CONFIGURATION (Florence-2)
# -------------------------------------------------
# We use 'cuda' if you have an NVIDIA GPU, otherwise 'cpu' (slower but works)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"
# Pin HF revision to avoid unexpected code updates
REVISION = "21a599d414c4d928c9032694c424fb94458e3594"

print(f"Loading the Teacher Model ({MODEL_ID})... this may take a minute...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    attn_implementation="eager",
    revision=REVISION,
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)

# Define your classes strictly
# Prompt: The text description the AI looks for
# ID: The class ID for YOLO (0 or 1)
TARGETS = [
    {"prompt": "orange cat", "id": 0, "name": "Simba"},
    {"prompt": "black cat",  "id": 1, "name": "Nala"}
]

def run_florence_inference(image_pil, text_prompt):
    """Asks the 'Smart Model' to find the object."""
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    # Florence-2 expects task token + input text for phrase grounding
    full_text = f"{task_prompt}{text_prompt}"
    results = processor(text=full_text, images=image_pil, return_tensors="pt")
    
    input_ids = results["input_ids"].to(DEVICE)
    pixel_values = results["pixel_values"].to(DEVICE)

    # Generate answer
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        use_cache=True,
        num_beams=1
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse the answer into coordinates
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image_pil.width, image_pil.height)
    )
    return parsed_answer[task_prompt]

def process_videos():
    video_files = sorted(list(Path(INPUT_VIDEO_FOLDER).glob("*.mp4")))
    total_samples = 0
    total_videos = len(video_files)
    
    print(f"Starting Auto-Labeling on {total_videos} videos...")

    for video_idx, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 2) # Capture 1 frame every 2 seconds (Diversity > Quantity)
        
        video_progress = (video_idx - 1) / total_videos * 100
        print(f"\n[Video {video_idx}/{total_videos} - {video_progress:.1f}%] Processing: {video_path.name}")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Skip frames to get diversity
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            # Convert to PIL for the Model
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = image_pil.size
            
            yolo_labels = []
            has_detection = False
            
            # Ask the Teacher Model for each cat separately
            for target in TARGETS:
                # We combine the prompt to be specific
                full_prompt = target["prompt"]
                prediction = run_florence_inference(image_pil, full_prompt)
                
                # Prediction format: {'bboxes': [[x1, y1, x2, y2]], 'labels': ['orange cat']}
                bboxes = prediction.get('bboxes', [])
                
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to YOLO (Normalized Center-X, Center-Y, Width, Height)
                    b_cx = ((x1 + x2) / 2) / w
                    b_cy = ((y1 + y2) / 2) / h
                    b_w = (x2 - x1) / w
                    b_h = (y2 - y1) / h
                    
                    # Store Label
                    yolo_labels.append(f"{target['id']} {b_cx:.6f} {b_cy:.6f} {b_w:.6f} {b_h:.6f}")
                    has_detection = True
                    
                    # Optional: Visual Debug to screen (slows down process slightly)
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            
            # Save Data if we found something
            if has_detection:
                base_name = f"{video_path.stem}_{frame_idx}"
                
                # 1. Save Image
                cv2.imwrite(os.path.join(IMAGES_DIR, f"{base_name}.jpg"), frame)
                
                # 2. Save Label Text
                with open(os.path.join(LABELS_DIR, f"{base_name}.txt"), "w") as f:
                    f.write("\n".join(yolo_labels))
                
                total_samples += 1
                percent_complete = (video_idx / total_videos) * 100
                print(f"Captured Sample #{total_samples}: {video_path.name} (Time: {frame_idx/fps:.1f}s) [{percent_complete:.1f}%]")
            
            frame_idx += 1
            
        cap.release()

    print(f"------------------------------------------------")
    print(f"Distillation Complete. {total_samples} labelled images generated.")
    print(f"Please open '{IMAGES_DIR}' to inspect them.")

if __name__ == "__main__":
    process_videos()