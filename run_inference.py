from ultralytics import YOLO
import os
import logging
from pathlib import Path

# Suppress verbose logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FFREPORT"] = "file=/dev/null"  # Suppress FFmpeg warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# ================= SILENCER CLASS =================
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

def run_inference():
    # Load the trained model
    model_path = r"runs\detect\cat_detector_v15\weights\best.pt"
    model = YOLO(model_path)
    model.overrides['verbose'] = False  # Suppress YOLO verbose output
    
    # Path to videos
    video_folder = Path("merged_videos")
    output_folder = Path("inference_results")
    output_folder.mkdir(exist_ok=True)
    
    # Get all video files
    video_files = list(video_folder.glob("*.mp4"))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        # Run inference silently with stderr suppressed
        with SuppressStderr():
            results = model.predict(
                source=str(video_path),
                stream=True,
                save=True,
                conf=0.25,
                iou=0.45,
                device='cpu',
                project='inference_results',
                name=video_path.stem,
                exist_ok=True,
                verbose=False  # Suppress frame-by-frame output
            )
            
            # Iterate over results to trigger inference
            for _ in results:
                pass  # Process silently
        
        print(f"✓ Completed: {video_path.name}")
    
    print(f"\n{'='*50}")
    print(f"All videos processed!")
    print(f"Results saved to: {output_folder.absolute()}")
    print(f"{'='*50}")

if __name__ == '__main__':
    run_inference()