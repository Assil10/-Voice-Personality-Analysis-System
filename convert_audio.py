import sys
print("Python executable:", sys.executable)
from moviepy import AudioFileClip
import os

input_folder = 'mp4_raw'
output_folder = 'audio_files'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".mp4", ".wav"))
        audio = AudioFileClip(input_path)
        audio.write_audiofile(output_path)
