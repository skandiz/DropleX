import os
import requests

# Define files and their corresponding download URLs
FILES = {
    "video_input/sample_video.mp4": "https://github.com/skandiz/DropleX/releases/download/v1.0/sample_video.mp4",
    "tracking_results/sample_video/skandiz_model_rgb/raw_detection_0_35999.parquet": "https://github.com/skandiz/DropleX/releases/tag/v1.0/raw_detection_0_35999.parquet",
    "tracking_results/sample_video/skandiz_model_rgb/raw_tracking_0_35999.parquet": "https://github.com/skandiz/DropleX/releases/tag/v1.0/raw_tracking_0_35999.parquet",
    "tracking_results/sample_video/skandiz_model_rgb/interpolated_tracking_0_35999.parquet": "https://github.com/skandiz/DropleX/releases/tag/v1.0/interpolated_tracking_0_35999.parquet",
    "tracking_results/sample_video/skandiz_model_rgb/trajectories_kalman_rts_0_35999_subsample_factor_1.parquet": "https://github.com/skandiz/DropleX/releases/tag/v1.0/trajectories_kalman_rts_0_35999_subsample_factor_1.parquet"
}

# Ensure both directories exist
for filepath in FILES.keys():
    dest_dir = os.path.dirname(filepath)
    os.makedirs(dest_dir, exist_ok=True)

# Download files if they don’t already exist
for filepath, url in FILES.items():
    if not os.path.exists(filepath):
        # Send request with proper headers and allow redirects
        response = requests.get(url, stream=True, headers={'Accept': 'application/octet-stream'}, allow_redirects=True)

        # Check if the request was successful
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0)) / (1024 * 1024)  # Convert to MB
            print(f"Downloading {filepath} ({total_size:.2f} MB)...")

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"{filepath} downloaded successfully.")
        else:
            print(f"Failed to download {filepath}. Status code: {response.status_code}")
            print("Response headers:", response.headers)

    else:
        print(f"{filepath} already exists.")