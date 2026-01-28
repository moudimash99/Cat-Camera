from ultralytics import YOLO
import time
from pathlib import Path
import os

# Prevent OpenMP runtime error on some Windows setups
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def benchmark_laptop():
    # 1. Setup
    model_path = r"runs\detect\cat_detector_v15\weights\best_openvino_model"
    # Update this to the name of one of your video files
    video_path = r"output\merged_videos\merged_005_20260128_105503.mp4" 
    
    print(f"--- Loading Model: {model_path} ---")
    model = YOLO(model_path)
    
    print(f"--- Starting Benchmark on: {Path(video_path).name} ---")
    print("Optimization: Streaming enabled (saves RAM)")
    
    # 2. Start Timer
    start_time = time.time()
    frame_count = 0
    
    # 3. Run Inference (Stream=True is critical for old laptops)
    results = model.predict(
        source=video_path,
        stream=True,     # Fixes memory crash
        save=True,       # Saves the video (set to False to test pure processing speed)
        device='cpu',    # Forces CPU mode
        imgsz=640,       # Default size. Change to 320 for 2x speed boost!
        conf=0.25,
        project='benchmark_results',
        name='laptop_test',
        exist_ok=True,
        verbose=False    # Hides the noisy default logs
    )

    # 4. Process Loop
    print("\nProcessing... (Press Ctrl+C to stop early)")
    try:
        for result in results:
            frame_count += 1
            
            # Print stats every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                # \r overwrites the line so your console doesn't get spammy
                print(f"\rFrames: {frame_count} | Speed: {current_fps:.1f} FPS", end="")
                
    except KeyboardInterrupt:
        print("\nTest stopped by user.")

    # 5. Final Stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\n\n{'='*30}")
    print(f"BENCHMARK COMPLETE")
    print(f"Total Frames: {frame_count}")
    print(f"Time Elapsed: {total_time:.1f} seconds")
    print(f"Average FPS:  {avg_fps:.2f}")
    print(f"{'='*30}")

if __name__ == '__main__':
    benchmark_laptop()