import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def show_all_plots(file_name, interval):
    # Load audio file
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 12))
    
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
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.title('Audio Signal')
    plt.tight_layout()

    # Plot pre-emphasis
    plt.figure(figsize=(8, 4))
    librosa.display.waveplot(y_preemp, sr=sr)
    plt.title('Pre-emphasis')
    plt.tight_layout()

    # Plot windowing
    plt.figure(figsize=(8, 4))
    plt.plot(window)
    plt.title('Windowing')
    plt.tight_layout()

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
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S, x_axis='time', y_axis='linear', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power Spectrum')
    plt.tight_layout()

    # Plot Mel filterbank
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_basis, x_axis='linear', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('Mel Filterbank')
    plt.tight_layout()

    # Plot logarithmic compression for all frames
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(log_mel_S_list, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Logarithmic Compression')
    plt.tight_layout()

    # Plot MFCCs for all frames
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('MFCCs')
    plt.tight_layout()

    # Show all plots
    plt.show()

if __name__ == '__main__':
    file_name = 'video.mp4'
    interval = 5
    show_all_plots(file_name, interval)