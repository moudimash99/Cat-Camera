import paramiko
import os
from datetime import datetime, timedelta
import stat
import subprocess
import glob

# ================= CONFIGURATION =================
# Server Credentials
HOST = "100.111.127.103"
USER = "moudimash99"
PASS = "mashaka99"
PORT = 22
file_path = "/home/moudimash99/Cat-Weight/data/cat_data.db"
# Remote Frigate Path (Where recordings are on the 192.168.1.135 server)
REMOTE_BASE_DIR = "/home/moudimash99/frigate/storage/recordings"

# Local Path (Where you want to save them on your computer)
LOCAL_DOWNLOAD_DIR = "./downloads"

# Time Range to Download
START_TIME_STR = "2026-01-18 01:30:00" 
END_TIME_STR   = "2026-01-18 2:15:00"
# =================================================

def download_clips():
    # 1. Establish SFTP Connection
    print(f"Connecting to {HOST}...")
    try:
        transport = paramiko.Transport((HOST, PORT))
        transport.connect(username=USER, password=PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 2. Prepare Local Directory
    if not os.path.exists(LOCAL_DOWNLOAD_DIR):
        os.makedirs(LOCAL_DOWNLOAD_DIR)

    # 3. Time Calculation Logic
    start_dt = datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M:%S")
    
    current_hour = start_dt.replace(minute=0, second=0, microsecond=0)

    print(f"Searching for videos between {START_TIME_STR} and {END_TIME_STR}...")

    download_count = 0

    # 4. Iterate through every hour in the range
    while current_hour <= end_dt:
        # Construct the remote path: YYYY-MM/DD/HH
        # Note: Linux paths always use forward slashes '/'
        remote_path = (f"{REMOTE_BASE_DIR}/"
                       f"{current_hour.strftime('%Y-%m')}-"
                       f"{current_hour.strftime('%d')}/"
                       f"{current_hour.strftime('%H')}/living_room")

        try:
            # List files in that remote directory
            file_list = sftp.listdir(remote_path)
            
            for filename in file_list:
                # We only want .mp4 files
                if not filename.endswith(".mp4"): continue

                # Check if file matches our time window (filename is MM.SS.mp4)
                try:
                    parts = filename.split('.')
                    mm = int(parts[0])
                    ss = int(parts[1])
                    
                    # Create the full timestamp for this specific file
                    file_time = current_hour.replace(minute=mm, second=ss)
                    
                    # STRICT CHECK: Is it in the window?
                    if start_dt <= file_time <= end_dt:
                        remote_full_path = f"{remote_path}/{filename}"
                        local_full_path  = os.path.join(LOCAL_DOWNLOAD_DIR, f"{file_time.strftime('%Y%m%d_%H%M%S')}.mp4")
                        
                        print(f"Downloading: {filename} -> {local_full_path}")
                        sftp.get(remote_full_path, local_full_path)
                        download_count += 1
                        
                except (ValueError, IndexError):
                    continue # Skip files that don't match expected format

        except IOError:
            # This happens if the folder for that hour doesn't exist on the server (no recordings)
            print(f"No recordings found for hour: {current_hour.strftime('%H:00')}")
            pass

        # Move to next hour
        current_hour += timedelta(hours=1)

    # 5. Cleanup
    sftp.close()
    transport.close()
    print("------------------------------------------------")
    print(f"Finished. Downloaded {download_count} files to {LOCAL_DOWNLOAD_DIR}")


# create a function that can copy a list of files from a source directory on ssh to local directory

def copy_files_from_ssh(file_list, local_dir):
    # Establish SFTP Connection
    print(f"Connecting to {HOST}...")
    try:
        transport = paramiko.Transport((HOST, PORT))
        transport.connect(username=USER, password=PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Prepare Local Directory
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for remote_file in file_list:
        filename = os.path.basename(remote_file)
        local_file_path = os.path.join(local_dir, filename)
        try:
            print(f"Copying: {remote_file} -> {local_file_path}")
            sftp.get(remote_file, local_file_path)
        except Exception as e:
            print(f"Failed to copy {remote_file}: {e}")

    # Cleanup
    sftp.close()
    transport.close()
    print("File copy operation completed.")

# Merge 10-second videos into 5-minute chunks
def merge_videos_into_chunks(input_dir, output_dir, chunk_duration_seconds=300):
    """
    Merge 10-second videos into 5-minute (300 second) chunks.
    
    Args:
        input_dir: Directory containing downloaded .mp4 files
        output_dir: Directory to save merged videos
        chunk_duration_seconds: Target duration for each merged video (default: 300 = 5 min)
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all mp4 files sorted by filename (which includes timestamp)
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos to merge")
    
    # Group videos into chunks (each chunk should have ~30 videos of 10s = 5 min)
    chunk_size = chunk_duration_seconds // 10  # 300 / 10 = 30 videos per chunk
    
    for chunk_idx, i in enumerate(range(0, len(video_files), chunk_size)):
        chunk_videos = video_files[i:i+chunk_size]
        
        # Create concat file for FFmpeg
        concat_file = os.path.join(output_dir, f"concat_{chunk_idx}.txt")
        with open(concat_file, 'w') as f:
            for video in chunk_videos:
                # Escape backslashes for FFmpeg on Windows
                escaped_path = video.replace('\\', '/')
                f.write(f"file '{escaped_path}'\n")
        
        # Output filename based on first video's timestamp
        first_video_name = os.path.basename(chunk_videos[0])
        output_video = os.path.join(output_dir, f"merged_{chunk_idx:03d}_{first_video_name}")
        
        print(f"\nMerging chunk {chunk_idx} ({len(chunk_videos)} videos) -> {output_video}")
        
        # Run FFmpeg to concatenate videos
        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # No re-encoding (fast)
                '-y',  # Overwrite output file
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully created {output_video}")
                # Clean up concat file
                os.remove(concat_file)
            else:
                print(f"✗ FFmpeg error: {result.stderr}")
        
        except FileNotFoundError:
            print("ERROR: FFmpeg not found. Install it with: choco install ffmpeg")
            return
        except Exception as e:
            print(f"Error merging chunk {chunk_idx}: {e}")

if __name__ == "__main__":
    # download_clips()
    copy_files_from_ssh([file_path], LOCAL_DOWNLOAD_DIR)
    
    # Uncomment to merge downloaded videos into 5-minute chunks:
    # merge_videos_into_chunks(LOCAL_DOWNLOAD_DIR, "./merged_videos")