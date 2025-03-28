import os
import sys
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
import numpy as np

def get_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    parsed_url = urlparse(youtube_url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")
    return None

def download_audio(youtube_url, output_dir, output_file="audio.wav"):
    """Download audio and thumbnail from a YouTube video and save them in the specified directory."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Construct the full path for the output audio file
    output_path = os.path.join(output_dir, output_file)

    # Adjust the yt-dlp output template to ensure files are saved in the correct directory
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Download the audio and thumbnail using yt-dlp
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",  # Extract audio as WAV
        "--write-thumbnail",  # Download the thumbnail
        # "--convert-thumbnails", "webp",  # Convert the thumbnail to WEBP format
        "-o", output_template,  # Save audio and thumbnail in the specified directory
        youtube_url
    ]
    subprocess.run(command, check=True)

    # Rename the downloaded WAV file to match the desired output_file name
    downloaded_audio_path = os.path.splitext(output_template)[0] + ".wav"
    if os.path.exists(downloaded_audio_path):
        os.rename(downloaded_audio_path, output_path)
        print(f"Renamed audio file to {output_path}")

    # Construct the expected thumbnail path
    thumbnail_path = os.path.splitext(output_template)[0] + ".webp"

    # Check if the thumbnail exists
    if os.path.exists(thumbnail_path):
        print(f"Thumbnail downloaded to {thumbnail_path}")
    else:
        print("Thumbnail not found. Please check if the video has a thumbnail available.")

    return output_path, thumbnail_path

def generate_waveform(audio_file, output_image_prefix="waveform", num_lines=600, total_width=10):
    """Generate 4 waveform images, each representing a quarter of the waveform, and save them in a named directory."""
    # Create a directory based on the output_image_prefix
    output_dir = output_image_prefix  # Use the prefix directly as the directory name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Check if the audio data is valid
    if not isinstance(y, np.ndarray) or y.size == 0:
        raise ValueError(f"Invalid audio data in file: {audio_file}")

    # Downsample the waveform to match the desired number of vertical lines
    step = max(1, len(y) // num_lines)  # Calculate step size to reduce data points
    y_downsampled = y[::step]  # Downsample the waveform

    # Normalize the downsampled waveform to fit within the image height
    y_normalized = y_downsampled / np.max(np.abs(y_downsampled))  # Normalize between -1 and 1

    # Split the waveform into 4 equal parts
    quarter_length = len(y_normalized) // 4
    quarters = [
        y_normalized[:quarter_length],
        y_normalized[quarter_length:2 * quarter_length],
        y_normalized[2 * quarter_length:3 * quarter_length],
        y_normalized[3 * quarter_length:]
    ]

    # Generate and save each quarter as a separate image
    for i, quarter in enumerate(quarters):
        # Calculate the spacing between lines based on the total width
        line_spacing = total_width / (len(quarter) - 1)

        # Create the plot
        plt.figure(figsize=(total_width, 4))  # Set the figure size dynamically

        # Plot each vertical line symmetrically
        for j, value in enumerate(quarter):
            x = j * line_spacing  # Calculate the x-coordinate for each line
            plt.plot([x, x], [-abs(value), abs(value)], color="#4d77ed", linewidth=2)  # Draw a vertical line reflected on both sides

        # Customize the plot
        plt.axis("off")  # Remove axis labels and ticks
        plt.tight_layout(pad=0)  # Remove padding around the plot

        # Save the image with a transparent background
        output_image = os.path.join(output_dir, f"{output_image_prefix}_{i + 1}.png")
        plt.savefig(output_image, bbox_inches="tight", pad_inches=0, dpi=300, facecolor="none")
        plt.close()
        print(f"Waveform quarter saved to {output_image}")

def is_valid_youtube_url(url):
    """Check if the provided URL is a valid YouTube URL."""
    return "youtube.com/watch" in url or "youtu.be/" in url

def main():
    # Check if a URL is passed as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python3 generate_waveform.py <YouTube Video URL>")
        sys.exit(1)

    youtube_url = sys.argv[1]

    # Validate the YouTube URL
    if not is_valid_youtube_url(youtube_url):
        print("Error: Invalid YouTube URL.")
        sys.exit(1)

    # Extract video ID and construct file names
    video_id = get_video_id(youtube_url)
    if not video_id:
        print("Error: Could not extract video ID from the URL.")
        sys.exit(1)

    audio_file = f"{video_id}.wav"
    waveform_image_prefix = f"{video_id}_waveform"

    # Check if the WAV file already exists
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found. Downloading...")
        download_audio(youtube_url, video_id, audio_file)
    else:
        print(f"Using existing audio file: {audio_file}")

    # Generate waveform
    generate_waveform(audio_file, waveform_image_prefix)

if __name__ == "__main__":
    main()