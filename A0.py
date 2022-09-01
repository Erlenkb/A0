import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import values


array = [1,2,2,1,34,1,3,2,34,1,2,2,21,3,2]


def _nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def _octave_filter():



    return 1

def _fft_signal(array, fs):
    fft = np.fft.fft(array, fs)
    freq = np.fft.fftfreq(n=len(fft), d=1/fs)
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), color="blue", label="Upper microphone")
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    #plt.xlim(82, 2000)
    #plt.ylim(15, 50)
    plt.legend()

    plt.show()



def _calculate_Lp(array):
    rms_pressure = np.sqrt(np.mean(array**2))
    return 20*np.log10(rms_pressure / values.p0)

def _calc_p_peak(cal_value):
    p_rms = values.p0 * 10**(cal_value/20)
    p_peak = p_rms / 0.707
    print("Pressure peak: {}".format(p_peak))
    return round(p_peak,9)

def _scaling_factor(cal_array):
    p_peak = _calc_p_peak(values.cal_value)
    d_peak = np.mean(np.sort(cal_array[len(cal_array) // 4 : len(cal_array)-len(cal_array) // 4])[-10:])
    print("Discrete peak: {}".format(d_peak))
    return d_peak / p_peak

def _scale_array(array, cal_array):
    A = _scaling_factor(cal_array)    
    return array / A

def _check_calibration(cal_before, cal_after):
    cal_before = _scale_array(cal_before, cal_before)
    cal_after = _scale_array(cal_after, cal_after)

    Lp_before = _calculate_Lp(cal_before)
    Lp_after = _calculate_Lp(cal_after)

    print("Calibration before measurements: {0:.1f} dB\nCalibration after measurements: {1:.1f} dB ".format(Lp_before,Lp_after))








if __name__ == '__main__':
    
    fs_sound_file, sound_file = wavfile.read("A0_lydfil.wav")
    fs_cal, cal_before = wavfile.read("cal_before.wav")
    fs_cal, cal_after = wavfile.read("cal_after.wav")

    #print("A:", _scaling_factor(cal_before))

    _check_calibration(cal_before,cal_after)
    array = _scale_array(sound_file,cal_before)
    fft = np.fft.fft(array, fs_cal)
    fft = 20*np.log10(np.abs(fft) / values.p0)
    freq = np.fft.fftfreq(n=len(fft), d=1/fs_cal)
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), color="blue", label="Upper microphone")
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    
    plt.legend()

    plt.show()

    print(_calc_p_peak(94))

    #print(_calculate_Lp(sound_file))