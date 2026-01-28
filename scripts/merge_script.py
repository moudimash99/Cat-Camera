import cv2
import os
from pathlib import Path

def merge_videos(input_folder, output_path):
    # Recursively find all video files in subdirectories
    video_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    
    video_paths = sorted(video_paths)
    
    if not video_paths:
        print("No video files found")
        return
    
    print(f"Found {len(video_paths)} video files to merge")
    
    first_video = cv2.VideoCapture(video_paths[0])
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for video_path in video_paths:
        print(f"Processing: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
    
    out.release()
    print(f"Merged video saved to {output_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\Theotime\Documents\GitHub\Cat-Camera\output\inference_results"
    output_path = r"C:\Users\Theotime\Documents\GitHub\Cat-Camera\output\merged_video.mp4"
    merge_videos(input_folder, output_path)