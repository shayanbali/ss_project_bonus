import glob

import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import fft
from scipy.io import wavfile
import os
import librosa


def powerSpectrum(address):
    sampling_rate, signal = wavfile.read(address)
    signal = signal.sum(axis=1)
    fourier_transform = np.fft.rfft(signal)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)
    frequency = fft.fftfreq(len(signal), 1 / sampling_rate)[:len(power_spectrum)]
    plt.plot(frequency, power_spectrum)
    # plt.plot(frequency, signal)
    plt.xlabel("freq")
    plt.ylabel(address.replace('.wav', ''))
    plt.xlim([0, 1000])
    plt.show()
    return frequency, power_spectrum


def findMax(address):
    sampling_rate, signal = wavfile.read(address)
    signal = signal.sum(axis=1)
    fourier_transform = np.fft.rfft(signal)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)
    frequency = fft.fftfreq(len(signal), 1 / sampling_rate)[:len(power_spectrum)]
    max_value = max(power_spectrum)
    power_spectrum = list(power_spectrum)
    max_index = power_spectrum.index(max_value)
    return frequency[max_index]


def folder_detection(directory):
    dict_label = {}
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        powerSpectrum(filename)
        print(str(filename) + " : " + str(findMax(filename)))
        if findMax(filename) < 180:
            dict_label[filename] = "male"
        else:
            dict_label[filename] = "female"
    return dict_label


def plot_signals_time(address):
    sampling_rate, signal = wavfile.read(address)
    signal = signal.astype(np.float32)
    time = np.linspace(0, signal.shape[0] / sampling_rate, signal.shape[0])
    plt.plot(time, signal)
    plt.xlabel("time")
    plt.ylabel(address.replace('.wav', ''))
    plt.show()


def AWGN(address, SNR):
    sampling_rate, signal = wavfile.read(address)
    signal = signal.astype(np.float32)
    rms_signal = math.sqrt(np.mean(signal ** 2))
    rms_noise = math.sqrt(rms_signal ** 2 / (pow(10, SNR / 10)))
    noise = np.random.normal(0, rms_noise, signal.shape[0]).astype(np.float32)
    r = signal + noise
    wavfile.write('noisy_' + address, sampling_rate, r.astype(np.int16))
    return r, signal, noise



def spectralSubtraction(address, noise):
    sampling_rate, noisy_signal = wavfile.read(address)
    noisy_signal = noisy_signal.astype(np.float32)
    fft_noisy_signal = librosa.stft(noisy_signal)
    abs_fft_noisy_signal = np.abs(fft_noisy_signal)

    abs_fft_noise = np.abs(librosa.stft(noise))
    fft_noise_avg = np.mean(abs_fft_noise, axis=1)

    angle_noisy = np.angle(fft_noisy_signal) * 1j
    exp_noisy = np.exp(angle_noisy)

    denoised_signal = librosa.istft((abs_fft_noisy_signal - fft_noise_avg.reshape((fft_noise_avg.shape[0], 1))) * exp_noisy).astype(np.int16)

    wavfile.write('denoised_' + address, sampling_rate, denoised_signal)


if __name__ == "__main__":
    # powerSpectrum('v1.wav')
    # print(findMax('v1.wav'))
    # folder_detection("voices")
    # print(folder_detection("voices"))
    # plot_signals_time("v0.wav")
    noisy_signal, signal, noise = AWGN('Test.wav', 10)
    plot_signals_time('Test.wav')
    plot_signals_time('noisy_Test.wav')
    spectralSubtraction('noisy_Test.wav', noise)
    plot_signals_time('denoised_noisy_Test.wav')

