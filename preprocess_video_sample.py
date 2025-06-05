from google.cloud import storage
import os
import cv2
import numpy as np
import librosa
import ffmpeg  # NEW: replacing moviepy

# Set credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\neela\Downloads\linear-theater-461715-m2-8e6e55f3ca64.json"

# GCS access
client = storage.Client()
bucket = client.bucket("ucftoy-dataset")

# List files with numbering
for i, blob in enumerate(bucket.list_blobs(prefix="hiphop_dataset/"), start=1):
    print(f"{i}. {blob.name}")

first_video_blob = next(
    (blob for blob in bucket.list_blobs(prefix="hiphop_dataset/") if blob.name.endswith(".webm")),
    None
)

if not first_video_blob:
    raise ValueError("No .webm video found in the bucket.")

os.makedirs("data", exist_ok=True)
local_path = os.path.join("data", "first_video.webm")
first_video_blob.download_to_filename(local_path)

print(f"Downloaded {first_video_blob.name} to {local_path}")


def extract_frames(video_path, num_frames=16, size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frames.append(frame)

    cap.release()
    return np.stack(frames) if len(frames) == num_frames else None


frames = extract_frames(local_path)
print(f"Frames shape: {frames.shape}")


def extract_mel(video_path, sr=22050, n_mels=64):
    wav_path = video_path.replace(".webm", ".wav")

    # NEW: extract audio using ffmpeg
    ffmpeg.input(video_path).output(wav_path, ac=1, ar=sr).run(quiet=True, overwrite_output=True)

    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


mel = extract_mel(local_path)
print(f"Mel spectrogram shape: {mel.shape}")

np.savez("data/first_sample.npz", frames=frames, mel=mel)
print("Saved to data/first_sample.npz")
