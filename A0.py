from tkinter import N
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import values

def _time_to_discrete(timeval,fs):
    return int(timeval * fs)

def _Leq_125ms(array,fs):
    N = _time_to_discrete(0.125, fs)
    it = len(array) // N
    array = np.split(array, it)
    l_eq = []
    for n in array: 
        sum = 0
        for i in n: sum += (i/values.p0)**2
        l_eq.append(10*np.log10(sum /len(array)))
    return l_eq

def _Leq_1s(array, fs):
    N = _time_to_discrete(1, fs)
    it = len(array) // N
    array = np.split(array, it)
    l_eq = []
    for n in array: 
        sum = 0
        for i in n: sum += (i/values.p0)**2
        l_eq.append(10*np.log10(sum /len(array)))
    return l_eq

def _nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def _octave_filter():



    return 1

def _fft_signal(array, fs):
    N = len(array)
    y = np.fft.fft(array,norm="backward")[0:int(N/2)]/N
    y[0:] = 2*y[0:]
    p_y = 20*np.log10(np.abs(y) / values.p0)
    f = fs*np.arange((N/2))/N
    sum = 0
    for i in p_y :
        sum += 10**(i/10)
    print("Lp from fft: {}".format(10*np.log10(sum)))
    fft = np.fft.fft(array,fs, norm="ortho")
    fft = 20*np.log10(fft/values.p0)
    freq = np.fft.fftfreq(n=len(fft), d=1/fs)
    plt.plot(f, p_y, color="blue", label="Upper microphone")
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(32.5, 5000)
    #plt.ylim(15, 50)
    plt.legend()
    plt.show()
    return freq, fft

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


def _plot_step_Leq(arr1, arr2, fs):
    time = len(arr1) / fs
    timeval = np.linspace(0,time, len(array))
    fig, ax = plt.subplots(figsize=(8,7))

    ax.step(timeval, )
    


if __name__ == '__main__':
    
    fs_sound_file, sound_file = wavfile.read("A0_lydfil.wav")
    fs_cal, cal_before = wavfile.read("cal_before.wav")
    fs_cal, cal_after = wavfile.read("cal_after.wav")

    


    #print("A:", _scaling_factor(cal_before))

    _check_calibration(cal_before,cal_after)
    array = _scale_array(sound_file,cal_before)
    
    freq, fft = _fft_signal(array, fs_cal)
    print("Lp signal:",_calculate_Lp(array))
    
    
    l_eq_125ms = _Leq_125ms(array,fs_cal)
    l_eq_1s = _Leq_1s(array,fs_cal)




  

    print(_calc_p_peak(94))
