##### This piece of code applies a band pass filter to an audio file. 

##### We set the low pass filter to 0 and we are only using the high pass filter. 

##### For more information about low pass filters here; https://en.wikipedia.org/wiki/Low-pass_filter

##### For more information about high pass filters here; https://en.wikipedia.org/wiki/High-pass_filter

##### For more information about band pass filters here; https://en.wikipedia.org/wiki/Band-pass_filter

##### Code has been modified from the original; https://americodias.com/docs/python/audio_python.md

##### The high pass filter is using spectral reversal. More about spectral reversal here; https://tomroelandts.com/articles/spectral-reversal-to-create-a-high-pass-filter

##### and here; https://tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter

##### Stelios Giannoulis, September 2020

import matplotlib.pyplot as plot
from scipy.io import wavfile
import scipy.signal as sg
from scipy.io.wavfile import write
import numpy as np
import contextlib
import wave

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('add your wav file here.wav')
samplingFrequency

def plot_audio_samples(title, samples, sampleRate, tStart=None, tEnd=None):
    if not tStart:
        tStart = 0

    if not tEnd or tStart>tEnd:
        tEnd = len(samples)/sampleRate

    f, axarr = plot.subplots(2, sharex=True, figsize=(20,10))
    axarr[0].set_title(title)
    axarr[0].plot(np.linspace(tStart, tEnd, len(samples)), samples)
    axarr[1].specgram(samples, Fs=sampleRate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(tStart,tEnd))
    #get_specgram(axarr[1], samples, sampleRate, tStart, tEnd)

    axarr[0].set_ylabel('Amplitude')
    axarr[1].set_ylabel('Frequency [Hz]')
    plot.xlabel('Time [sec]')

    plot.show()

def fir_high_pass(samples, fs, fH, N, outputType):
    # Referece: https://fiiir.com

    fH = fH / fs

    # Compute sinc filter.
    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.))
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[int((N - 1) / 2)] += 1
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

def fir_low_pass(samples, fs, fL, N, outputType):
    # Referece: https://fiiir.com

    fL = fL / fs

    # Compute sinc filter.
    h = np.sinc(2 * fL * (np.arange(N) - (N - 1) / 2.))
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

def convert_to_mono(channels, nChannels, outputType):
    if nChannels == 2:
        samples = np.mean(np.array([channels[0], channels[1]]), axis=0)  # Convert to mono
    else:
        samples = channels[0]

    return samples.astype(outputType)

def extract_audio(fname, tStart=None, tEnd=None):
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        startFrame, endFrame, segFrames = get_start_end_frames(nFrames, sampleRate, tStart, tEnd)

        # Extract Raw Audio from multi-channel Wav File
        spf.setpos(startFrame)
        sig = spf.readframes(segFrames)
        spf.close()

        channels = interpret_wav(sig, segFrames, nChannels, ampWidth, True)

        return (channels, nChannels, sampleRate, ampWidth, nFrames)
def get_start_end_frames(nFrames, sampleRate, tStart=None, tEnd=None):

    if tStart and tStart*sampleRate<nFrames:
        start = tStart*sampleRate
    else:
        start = 0

    if tEnd and tEnd*sampleRate<nFrames and tEnd*sampleRate>start:
        end = tEnd*sampleRate
    else:
        end = nFrames

    return (start,end,end-start)
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

tStart=0
tEnd=len(signalData)/samplingFrequency
tEnd

channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('SG2sZKL8_2019-09-14-05-54-17-754_2019-09-14-05-56-17-750_153.wav', tStart, tEnd)

#Convert audio to mono. No issue here. Audio files are already mono
samples = convert_to_mono(channels, nChannels, np.int16)

# Plot the original audio file and its spectrogram
plot_audio_samples("SG2sZKL8_2019-09-14-05-54-17-754_2019-09-14-05-56-17-750_153.wav", samples, sampleRate, tStart, tEnd)


# Apply the low pass filter. As we don't care about the low pass filter, we set it to 0. 
lp_samples_filtered = fir_low_pass(samples, sampleRate, 0, 461, np.int16)               # First pass
lp_samples_filtered = fir_low_pass(lp_samples_filtered, sampleRate, 0, 461, np.int16)   # Second pass

# Apply the high pass filter. We set the 3rd argument to 50, 75, 100,125, 150, 175, 200, which means it passes
# everything above 50Hz, 75Hz etc. etc.
hp_samples_filtered = fir_high_pass(samples, sampleRate,175, 461, np.int16)             # First pass
hp_samples_filtered = fir_high_pass(hp_samples_filtered, sampleRate,175, 461, np.int16) # Second pass

samples_filtered = np.mean(np.array([lp_samples_filtered, hp_samples_filtered]), axis=0).astype(np.int16)

#Plot the new audio file and its spectogram with the high pass filter applied. 
plot_audio_samples("SG2sZKL8_2019-09-14-05-54-17-754_2019-09-14-05-56-17-750_153.wav", samples_filtered, sampleRate, tStart, tEnd)

#Create the new audio file with the high pass filter applied.
wavfile.write('High_Pass_Filter_175Hz_SG2sZKL8_2019-09-14-05-54-17-754_2019-09-14-05-56-17-750_153.wav', sampleRate, samples_filtered)


