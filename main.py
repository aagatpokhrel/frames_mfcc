import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def video_to_image(video_file):
    video = cv2.VideoCapture(video_file)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    extract_frame = int(frame_rate * 5)

    for i in range(0, total_frames, extract_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read() 
        if success:
            cv2.imwrite("output/frame_{}.jpg".format(i), frame)
    video.release()

def extract_mfcc_features(video_file):
    y, sr = librosa.core.load(video_file)
    audio_length = librosa.get_duration(y=y, sr=sr)
    intervals = int(np.ceil(audio_length / 5))
    interval_start = 0
    mfccs = []
    for i in range(intervals):
        interval_end = min(interval_start + 5 * sr, len(y))
        interval = y[interval_start:interval_end]
        mfccs.append(librosa.feature.mfcc(interval, sr=sr))
        interval_start = interval_end
    return mfccs, sr

mfccs,sr = extract_mfcc_features("video.mp4")

plt.figure(figsize=(10, 4))
for i, mfcc in enumerate(mfccs):
    plt.subplot(len(mfccs), 1, i+1)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.title(f"MFCC for Frame {i+1}")
plt.tight_layout()
plt.show()