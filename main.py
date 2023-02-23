import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

def video_to_image(video_file, interval):
    video = cv2.VideoCapture(video_file)  #read video file

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # count the no. of frame in a video
    frame_rate = video.get(cv2.CAP_PROP_FPS) # get the frame rate of the video

    extract_frame = int(frame_rate * interval)

    for i in range(0, total_frames, extract_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)  # 0-based index of the frame to be decoded/captured next.
        success, frame = video.read() 
        if success:
            cv2.imwrite("output/frame_{}.jpg".format(i), frame)
    video.release()

def show_mfcc_plots_by_frames(file_name, interval):
    y, sr = librosa.core.load(file_name) # Load an audio file as a floating point time series.
    audio_length = librosa.get_duration(y=y, sr=sr)  # compute the duration (in seconds) of an audio time series
    intervals = int(np.ceil(audio_length / 5))  #Return the ceiling of the input, element-wise. eg a = [-1.3, -1.2, -1] returns [-1, -1, -1]
    interval_start = 0
    mfccs = []
    for i in range(intervals):
        interval_end = min(interval_start + 5 * sr, len(y))
        interval = y[interval_start:interval_end]
        mfccs.append(librosa.feature.mfcc(y=interval, sr=sr)) # mfccs feature extraction
        interval_start = interval_end
    
    plt.figure(figsize=(16, 9))
    rows = int(math.ceil(len(mfccs)/2))
    for i, mfcc in enumerate(mfccs):
        plt.subplot(rows, 2, i+1) #ploting multiple plot in one figure
        librosa.display.specshow(mfcc, sr=sr, x_axis='time') # displays the feature spectrum
        plt.title(f"MFCC for Frame {i+1}")
    plt.subplots_adjust(hspace=1)
    # manage padding across subplots
    plt.show()


def show_all_plots(file_name, interval):
    # Load audio file
    plt.figure(figsize=(16, 9))
    y, sr = librosa.load(file_name, sr=44100)

    # Pre-emphasis
    y_preemp = librosa.effects.preemphasis(y)
    # Framing
    n_fft = 2048
    hop_length = sr*interval
    y_frames = librosa.util.frame(y_preemp, frame_length=n_fft, hop_length=hop_length)
    print(len(y_frames))
    # Windowing
    window = np.hanning(n_fft)
    y_win = y_frames * window.reshape((-1, 1))
    print (len(y_win))
    # Plot audio signal
    plt.subplot(4, 2, 1) 
    # plt.figure(figsize=(8, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.title('Audio Signal')

    # Plot pre-emphasis
    plt.subplot(4, 2, 2) 
    # plt.figure(figsize=(8, 4))
    librosa.display.waveplot(y_preemp, sr=sr)
    plt.title('Pre-emphasis')

    # Plot windowing
    plt.subplot(4, 2, 3)
    # plt.figure(figsize=(8, 4))
    plt.plot(window)
    plt.title('Windowing')

    # Compute features for each frame
    n_mels = 40
    n_mfcc = 13
    mfccs = []
    log_mel_S_list = []
    S = np.zeros((n_fft // 2 + 1, y_win.shape[1]))
    for i in range(y_win.shape[1]):
        # Fourier transform
        Y = np.fft.rfft(y_win[:, i], n_fft, axis=0)
        # Power spectrum
        S[:, i] = np.abs(Y) ** 2
        # Mel filterbank
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        mel_S = np.dot(mel_basis, S[:, i])
        # Logarithmic compression
        log_mel_S = librosa.amplitude_to_db(mel_S)
        # MFCCs
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=n_mfcc)
        mfccs.append(mfcc)
        log_mel_S_list.append(log_mel_S)

    mfccs = np.array(mfccs)
    log_mel_S_list = np.array(log_mel_S_list)

    # Plot power spectrum
    # plt.subplot(4, 2, 4) 
    # # plt.figure(figsize=(8, 4))
    # librosa.display.specshow(S, x_axis='time', y_axis='linear', sr=sr, hop_length=hop_length)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Power Spectrum')
    plt.subplot(4,2,4)
    for i in range(mel_basis.shape[0]):
        plt.plot(librosa.util.normalize(mel_basis[i]))
    plt.title("Normalized Mel Filter")

    # Plot Mel filterbank
    plt.subplot(4, 2, 5) 
    # plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_basis, x_axis='linear', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('Mel Filterbank')

    # Plot logarithmic compression for all frames
    plt.subplot(4, 2, 6) 
    # plt.figure(figsize=(8, 4))
    librosa.display.specshow(log_mel_S_list, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Logarithmic Compression')

    # Plot MFCCs for all frames
    plt.subplot(4, 2, 7) 
    # plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('MFCCs')

    plt.subplots_adjust(hspace=1)
    # Show all plots
    plt.show()

if __name__ == '__main__':
    file_name = 'vid.mp4'
    interval = 5
    # video_to_image(file_name, interval)
    
    show_all_plots(file_name, interval)
    #show_mfcc_plots_by_frames(file_name, interval)
