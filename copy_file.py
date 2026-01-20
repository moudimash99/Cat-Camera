import paramiko
import os
from datetime import datetime, timedelta
import stat

# ================= CONFIGURATION =================
# Server Credentials
HOST = "192.168.1.135"
USER = "moudimash99"
PASS = "mashaka99"
PORT = 22

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

if __name__ == "__main__":
    download_clips()