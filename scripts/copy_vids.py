import paramiko
import os
from datetime import datetime, timedelta
import stat
import subprocess
import glob
import re

# ================= CONFIGURATION =================
# Server Credentials
HOST = "100.111.127.103"
USER = "moudimash99"
PASS = "mashaka99"
PORT = 22
file_path = "/home/moudimash99/Cat-Weight/data/cat_data.db"
# Remote Frigate Path (Where recordings are on the 192.168.1.135 server)
REMOTE_BASE_DIR = "/home/moudimash99/frigate/storage/recordings"

# Base downloads directory
DOWNLOADS_BASE_DIR = "./downloads"

# Time Range to Download
START_TIME_STR = "2026-01-28 10:30:00" 
END_TIME_STR   = "2026-01-28 11:15:00"
# =================================================

def get_next_download_folder():
    """
    Find the next available download folder number.
    Looks for existing downloads_X folders and returns downloads_{X+1}
    """
    if not os.path.exists(DOWNLOADS_BASE_DIR):
        os.makedirs(DOWNLOADS_BASE_DIR)
    
    # Find all existing numbered folders
    existing_folders = []
    for folder in os.listdir(DOWNLOADS_BASE_DIR):
        match = re.match(r'downloads_(\d+)', folder)
        if match:
            existing_folders.append(int(match.group(1)))
    
    # Get next number
    next_num = max(existing_folders) + 1 if existing_folders else 1
    
    return os.path.join(DOWNLOADS_BASE_DIR, f"downloads_{next_num}")

def setup_download_structure(base_folder):
    """
    Create the organized folder structure:
    - base_folder/10s_videos/
    - base_folder/5min_merged/
    - base_folder/full_merged/
    """
    subfolders = {
        '10s_videos': os.path.join(base_folder, '10s_videos'),
        '5min_merged': os.path.join(base_folder, '5min_merged'),
        'full_merged': os.path.join(base_folder, 'full_merged')
    }
    
    for folder_path in subfolders.values():
        os.makedirs(folder_path, exist_ok=True)
    
    return subfolders

def download_clips(local_download_dir):
    """
    Download clips from the remote server to the specified directory.
    
    Args:
        local_download_dir: Directory to save downloaded videos
    """
    # 1. Establish SFTP Connection
    print(f"Connecting to {HOST}...")
    try:
        transport = paramiko.Transport((HOST, PORT))
        transport.connect(username=USER, password=PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 0

    # 2. Prepare Local Directory
    if not os.path.exists(local_download_dir):
        os.makedirs(local_download_dir)

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
                        local_full_path  = os.path.join(local_download_dir, f"{file_time.strftime('%Y%m%d_%H%M%S')}.mp4")
                        
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
    print(f"Finished. Downloaded {download_count} files to {local_download_dir}")
    return download_count


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
        output_dir: Directory to save merged 5-minute videos
        chunk_duration_seconds: Target duration for each merged video (default: 300 = 5 min)
    
    Returns:
        Number of merged chunks created
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all mp4 files sorted by filename (which includes timestamp)
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return 0
    
    print(f"Found {len(video_files)} videos to merge into 5-minute chunks")
    
    # Group videos into chunks (each chunk should have ~30 videos of 10s = 5 min)
    chunk_size = chunk_duration_seconds // 10  # 300 / 10 = 30 videos per chunk
    
    chunks_created = 0
    
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
                chunks_created += 1
            else:
                print(f"✗ FFmpeg error: {result.stderr}")
        
        except FileNotFoundError:
            print("ERROR: FFmpeg not found. Install it with: choco install ffmpeg")
            return 0
        except Exception as e:
            print(f"Error merging chunk {chunk_idx}: {e}")
    
    return chunks_created

def merge_all_videos_into_one(input_dir, output_dir):
    """
    Merge all 5-minute chunks into one final full video.
    
    Args:
        input_dir: Directory containing 5-minute merged videos
        output_dir: Directory to save the final full merged video
    
    Returns:
        Path to the final merged video, or None if failed
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all mp4 files sorted by filename
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return None
    
    print(f"\nMerging {len(video_files)} chunks into final full video...")
    
    # Create concat file for FFmpeg
    concat_file = os.path.join(output_dir, "concat_full.txt")
    with open(concat_file, 'w') as f:
        for video in video_files:
            # Escape backslashes for FFmpeg on Windows
            escaped_path = video.replace('\\', '/')
            f.write(f"file '{escaped_path}'\n")
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video = os.path.join(output_dir, f"full_merged_{timestamp}.mp4")
    
    print(f"Creating full merged video: {output_video}")
    
    # Run FFmpeg to concatenate all videos
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
            print(f"✓ Successfully created full merged video: {output_video}")
            # Clean up concat file
            os.remove(concat_file)
            return output_video
        else:
            print(f"✗ FFmpeg error: {result.stderr}")
            return None
    
    except FileNotFoundError:
        print("ERROR: FFmpeg not found. Install it with: choco install ffmpeg")
        return None
    except Exception as e:
        print(f"Error merging full video: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING NEW DOWNLOAD RUN")
    print("=" * 60)
    
    # Step 1: Create new download folder structure
    download_folder = get_next_download_folder()
    print(f"\n📁 Creating download folder: {download_folder}")
    folders = setup_download_structure(download_folder)
    
    print(f"  ├─ 10s videos: {folders['10s_videos']}")
    print(f"  ├─ 5min merged: {folders['5min_merged']}")
    print(f"  └─ Full merged: {folders['full_merged']}")
    
    # Step 2: Download clips to 10s_videos folder
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING 10-SECOND VIDEOS")
    print("=" * 60)
    download_count = download_clips(folders['10s_videos'])
    
    if download_count == 0:
        print("\n⚠️  No videos downloaded. Exiting.")
        exit(0)
    
    # Step 3: Merge into 5-minute chunks
    print("\n" + "=" * 60)
    print("STEP 2: MERGING INTO 5-MINUTE CHUNKS")
    print("=" * 60)
    chunks_created = merge_videos_into_chunks(
        folders['10s_videos'], 
        folders['5min_merged']
    )
    
    if chunks_created == 0:
        print("\n⚠️  No chunks created. Exiting.")
        exit(0)
    
    print(f"\n✓ Created {chunks_created} 5-minute chunks")
    
    # Step 4: Merge all chunks into final video
    print("\n" + "=" * 60)
    print("STEP 3: CREATING FULL MERGED VIDEO")
    print("=" * 60)
    final_video = merge_all_videos_into_one(
        folders['5min_merged'],
        folders['full_merged']
    )
    
    if final_video:
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD RUN COMPLETE!")
        print("=" * 60)
        print(f"📁 Download folder: {download_folder}")
        print(f"📹 10-second videos: {download_count} files")
        print(f"🎬 5-minute chunks: {chunks_created} files")
        print(f"🎥 Full merged video: {final_video}")
        print("=" * 60)
    else:
        print("\n⚠️  Failed to create full merged video.")