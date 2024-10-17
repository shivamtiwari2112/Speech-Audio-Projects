
import numpy as np
from scipy.signal import find_peaks

def pitch_detector(signal):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(signal)-1:]

    peaks, _ = find_peaks(autocorr)
    max_peak = np.argmax(autocorr[peaks])
    pitch = peaks[max_peak]

    return pitch


def levDurbin(R):
    # Return PARCOR coefficients k and LPC coefficients a for input autocorrelation sequence R
    E = R[0]
    k = np.zeros(10)
    a = np.zeros(10)

    for i in range(1, 11):
        numer = R[i] - np.dot(a[:i][::-1], R[1:i+1])
        k[i-1] = numer / E
        a_temp = np.zeros_like(a)
        a_temp[:i] = a[:i] - k[i-1] * a[:i][::-1]
        a = a_temp.copy()
        a[i-1] = k[i-1]
        E = (1 - k[i-1]**2) * E

    return k, a

import scipy.io.wavfile as wav
from scipy.signal import medfilt
import time

def LPCencoder(signal):
    framelength = 180
    hamwin = np.hamming(framelength)
    zcr_thresh = 0.05

    # Speech Normalization
    signal = signal / np.max(np.abs(signal))
    amp = np.max(np.abs(signal))

    # Initial PCM
    quantizer = np.round((signal / amp) * 2047).astype(np.int16)
    signal = (quantizer / 2047) * amp

    # Initial HPF
    signal = np.convolve(signal, [1, -0.9375], mode='same')

    # Calculating number of windows
    numwindows = int(np.ceil(len(signal) / framelength))

    # Zero padding
    signal = np.append(signal, np.zeros(abs(len(signal) - numwindows * framelength)))

    # Initialize parameters to be transmitted
    period = np.zeros(numwindows)
    voiced = np.zeros(numwindows)
    gain = np.zeros(numwindows)
    coeff = np.zeros((numwindows, 10))

    total_processing_time = 0

    # Process Window by window
    for win in range(numwindows):
        frame = signal[win * framelength: (win + 1) * framelength]
        frame = frame * hamwin

        start_time = time.time()

        # LPC Coefficients
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(frame) - 1:]
        k, a = levDurbin(autocorr)
        k[:2] = np.log10((1 - k[:2]) / (1 + k[:2]))  # Convert to LAR
        coeff[win, :] = k

        # Zero Crossing Rate
        # Voiced/unvoiced detection through ZCR
        signmat = np.sign(frame)
        zcr = np.mean(np.abs(np.diff(signmat) / 2))
        if zcr > zcr_thresh:
            voiced[win] = 0
        else:
            voiced[win] = 1

        # Pitch Detection
        residual = np.convolve(frame, a, mode='same')
        if voiced[win]:
            # Impulse train generation
            residual = residual * np.hamming(len(residual))
            period[win] = pitch_detector(residual)

            # Set period within range of valid values
            if period[win] < 20:
                period[win] = 0
                voiced[win] = 0

            # Gain Calculation
            if period[win] == 0:
                gain[win] = np.sqrt(np.mean(residual**2))
            else:
                error = residual**2
                idx = int(np.floor(framelength / period[win])) * period[win]
                idx = int(idx)
                error = error[:idx]
                gaintemp = np.sum(error) / (idx)
                gain[win] = np.sqrt(period[win] * gaintemp)
        else:
            period[win] = 0
            gain[win] = np.sqrt(np.mean(residual**2))

        end_time = time.time()  # End time for processing the frame
        frame_processing_time = end_time - start_time  # Processing time for the current frame
        total_processing_time += frame_processing_time

    avg_processing_time_per_frame = total_processing_time / numwindows
    print("Average processing delay per frame is: {:.2f} ms".format(avg_processing_time_per_frame * 1000))

    # Median filtering of pitch periods
    period = medfilt(period, kernel_size=5)

    # Maximum gain calculation for quantization
    maxgain = np.max(gain)

    # Quantization
    # Gain Quantization to 5 bits
    gain_quant = np.round((gain / maxgain) * 31).astype(np.uint8)

    # Period Quantization to 6 bits
    period[(period < 42) & (period < 78)] = 2 * np.uint8(period[(period < 42) & (period < 78)] / 2)

    # LPC Coefficient Quantization
    lar_coeff = coeff[:, :2]
    lar_quant = np.round((lar_coeff / 2) * 15).astype(np.int8)

    par5_coeff = coeff[:, 2:4]
    par5_quant = np.round(par5_coeff * 15).astype(np.int8)

    par4_coeff = coeff[:, 4:8]
    par4_quant = np.round(par4_coeff * 7).astype(np.int8)

    par3_coeff = coeff[:, 8]
    par3_quant = np.round(par3_coeff * 3).astype(np.int8)

    par2_coeff = coeff[:, 9]
    par2_quant = np.round(par2_coeff * 1.5).astype(np.int8)

    coeff_quant = np.concatenate((lar_quant, par5_quant, par4_quant, par3_quant.reshape(-1, 1), par2_quant.reshape(-1, 1)), axis=1)

    parameters = np.concatenate((gain_quant.reshape(-1, 1), period.reshape(-1, 1), coeff_quant), axis=1)

    return parameters

# Load audio file
Fs, signal = wav.read('/content/footballisadangerousgame.wav')
signal = signal[:40000]  # Extract first 5 seconds of file

parameters = LPCencoder(signal)
np.savetxt('parameters.txt', parameters, fmt='%d')

total_bits = parameters.size * 8
print("Total Bits:", total_bits)

# %%
import numpy as np
from scipy.signal import lfilter

def LPCdecoder(parameters):
    # Load quantized parameters
    gain = parameters[:, 0]
    period = parameters[:, 1]
    coeff = parameters[:, 2:]

    speechlength = len(speech)
    framelength = 180
    numwindows = int(np.ceil(speechlength / framelength))
    reconstruct = np.zeros(speechlength)

    # Window by window reconstruction
    for j in range(numwindows):
        k = coeff[j]

        # LAR to PARCOR
        k[0:2] = (1 - 10**k[0:2].astype(float)) / (1 + 10**k[0:2].astype(float))

        # PARCOR to LPC Coefficients
        a = np.zeros(11)
        for i in range(1, 11):
            a[i] = k[i - 1]
            a[0:i] -= k[i - 1] * np.flipud(a[0:i])

        # Excitation Generation and Synthesis
        if period[j] > 0:
            impulse = np.zeros(int(period[j]))

            # Retain continuity between frames
            if period[j - 1] != 0 and j > 0:
                impulse[int(period[j] - lastpulse)] = 1
                lastpulse = int(period[j])
            else:
                impulse[0] = 1
                lastpulse = int(period[j])

            excitation = np.tile(impulse, 100)[:framelength]
            synth = gain[j] * lfilter([1], [1, *-a[1:]], excitation)
        else:
            excitation = np.random.randn(framelength)
            synth = gain[j] * lfilter([1], [1, *-a[1:]], excitation)

        # Adjust synth length to match framelength
        if (j + 1) * framelength > speechlength:
            synth = synth[:speechlength - j * framelength]

        reconstruct[j * framelength:(j + 1) * framelength] = synth

    # Post Filter
    reconstruct = lfilter([1], [1, -0.9375], reconstruct)

    return reconstruct


# Load quantized parameters
parameters = np.loadtxt('parameters.txt', dtype=np.int8)

# Load audio file
Fs, speech = wav.read('/content/footballisadangerousgame.wav')
speech = speech[:40000]  # Extract first 5 seconds of file

reconstruct = LPCdecoder(parameters)

# Save synthesized speech
wav.write('reconstructed_audio.wav', Fs, reconstruct.astype(np.int8))

# %%
from IPython.display import Audio, display

# Load the audio file
audio_path = '/content/reconstructed_audio.wav'

# Display the audio player
display(Audio(audio_path, autoplay=True))



