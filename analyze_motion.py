import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

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
MOTION_RESOLUTION = (640, 360)

# ================= 3. CORE FUNCTIONS =================
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

def analyze_videos():
    """
    Analyze motion scores for all videos and create visualizations
    """
    video_files = sorted(list(Path(INPUT_VIDEO_FOLDER).glob("*.mp4")))
    
    if not video_files:
        print(f"No video files found in {INPUT_VIDEO_FOLDER}")
        return
    
    print(f"Analyzing motion in {len(video_files)} videos...")
    
    # Store results for each video
    video_stats = []
    
    for video_idx, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[Video {video_idx}/{len(video_files)}] {video_path.name}")
        print(f"  FPS: {fps:.2f} | Total Frames: {total_frames}")
        
        prev_gray_small = None
        frame_idx = 0
        motion_scores = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate motion score
            motion_score, current_gray_small = get_motion_score(frame, prev_gray_small)
            prev_gray_small = current_gray_small
            
            motion_scores.append(motion_score)
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end="\r")
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate statistics for this video
        motion_array = np.array(motion_scores)
        stats = {
            'name': video_path.name,
            'scores': motion_scores,
            'mean': np.mean(motion_array),
            'median': np.median(motion_array),
            'std': np.std(motion_array),
            'min': np.min(motion_array),
            'max': np.max(motion_array),
            'p95': np.percentile(motion_array, 95),
            'p99': np.percentile(motion_array, 99),
        }
        video_stats.append(stats)
        
        print(f"\n  Stats: Mean={stats['mean']:.4f}% | Median={stats['median']:.4f}% | Max={stats['max']:.2f}%")
        print(f"         95th percentile={stats['p95']:.4f}% | 99th percentile={stats['p99']:.4f}%")
    
    # ================= 4. CREATE VISUALIZATIONS =================
    print("\n" + "="*60)
    print("Creating visualizations...")
    
    # Create figure with subplots for each video
    num_videos = len(video_stats)
    fig, axes = plt.subplots(num_videos, 2, figsize=(16, 5*num_videos))
    
    # Handle single video case (axes won't be 2D)
    if num_videos == 1:
        axes = axes.reshape(1, -1)
    
    for idx, stats in enumerate(video_stats):
        # Left plot: Motion score over time
        ax_time = axes[idx, 0]
        ax_time.plot(stats['scores'], linewidth=0.5, alpha=0.7)
        ax_time.axhline(y=stats['mean'], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.4f}%')
        ax_time.axhline(y=stats['p95'], color='orange', linestyle='--', label=f'95th: {stats["p95"]:.4f}%')
        ax_time.set_xlabel('Frame Number')
        ax_time.set_ylabel('Motion Score (%)')
        ax_time.set_title(f'{stats["name"]} - Motion Over Time')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # Right plot: Histogram of motion scores
        ax_hist = axes[idx, 1]
        ax_hist.hist(stats['scores'], bins=50, edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=stats['mean'], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.4f}%')
        ax_hist.axvline(x=stats['median'], color='g', linestyle='--', label=f'Median: {stats["median"]:.4f}%')
        ax_hist.axvline(x=stats['p95'], color='orange', linestyle='--', label=f'95th: {stats["p95"]:.4f}%')
        ax_hist.set_xlabel('Motion Score (%)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'{stats["name"]} - Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'motion_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    # ================= 5. SUMMARY STATISTICS =================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (All Videos)")
    print("="*60)
    
    for stats in video_stats:
        print(f"\n{stats['name']}:")
        print(f"  Mean:   {stats['mean']:.6f}%")
        print(f"  Median: {stats['median']:.6f}%")
        print(f"  Std:    {stats['std']:.6f}%")
        print(f"  Min:    {stats['min']:.6f}%")
        print(f"  Max:    {stats['max']:.6f}%")
        print(f"  95th:   {stats['p95']:.6f}%")
        print(f"  99th:   {stats['p99']:.6f}%")

# ================= 6. MAIN EXECUTION =================
if __name__ == "__main__":
    print("Motion Analysis Tool")
    print("Initializing... (FFmpeg warnings will be suppressed)")
    with SuppressStderr():
        analyze_videos()
