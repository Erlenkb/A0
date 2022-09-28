from tkinter import N
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import values

def _time_to_discrete(timeval,fs):
    return int(timeval * fs)

def _split_padded(a,n):
     padding = (-len(a))%n
     return np.split(np.concatenate((a,np.zeros(padding))),n)

def _Leq_125ms(array,fs):
    N = _time_to_discrete(0.125, fs)
    it = len(array) // N
    
    #print(it, " : ", len(array)," : ", len(array) / it)

    array1 = _split_padded(array,it)
    l_eq = []
    leq_total = 0
    for n in array1: 
        sum = 0
        for i in n: sum += (i/values.p0)**2
        l_eq.append(10*np.log10(sum /len(n)))
    time = np.linspace(0,len(array)/fs,it +1)
    for i in l_eq:
        leq_total += 10**(i/10)
    return l_eq, time, 10*np.log10(leq_total / len(l_eq))

def _Leq_1s(array, fs):
    N = _time_to_discrete(1, fs)
    it = len(array) // N

    #print(it, " : ", len(array)," : ", len(array) / it)

    array1 = _split_padded(array, it)
    
    l_eq = []
    leq_total = 0
    for n in array1: 
        sum = 0
        for i in n: sum += (i/values.p0)**2
        l_eq.append(10*np.log10(sum /len(n)))
    time = np.linspace(0,len(array)/fs,it + 1)
    for i in l_eq:
        leq_total += 10**(i/10)
    return l_eq, time, 10*np.log10(leq_total / len(l_eq))

def _nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def _Lp_from_freq(fft, freq):
    it = []
    for i, val in enumerate(freq):
        if (val >= 0 and val <= 20000):
            it.append(i)
    Lp = 0
    for i in it:
        Lp += 10**(fft[i]/10)
    return 10*np.log10(Lp)



def _fft_signal(array, fs):
    N = len(array)
    fft = np.fft.fft(array, fs)
    freq = np.fft.fftfreq(len(fft), d=1/fs)
    y = np.fft.fft(array)[0:int(N/2)]/N
    y[0:] = 2*y[0:]
    p_y = 20*np.log10(np.abs(y) / values.p0)
    f = fs*np.arange((N/2))/N
    sum = 0

    for i in p_y :
        sum += 10**(i/10)
    Lp = _Lp_from_freq(p_y, freq)
    print("Lp from fft: {}".format(Lp))
   
    plt.plot(f, np.abs(p_y), color="blue", label="FFT for sound signal with NFFT={} s".format(values.stop - values.start))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    plt.xlim(32.5, 12000)
    plt.legend()
    plt.show()
    return f, p_y

def _calculate_Lp(array):
    rms_pressure = np.sqrt(np.mean(array**2))
    return 20*np.log10(rms_pressure / values.p0)

def _calculate_Lp_freq(array):
    rms_pressure = np.sqrt(np.mean(array**2))
    return 0*np.log10(rms_pressure / values.p0)

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

def _plot_step_Leq(arr, fs):
    l_eq_125ms, time_125ms, leq_tot_125 = _Leq_125ms(arr,fs_cal)
    l_eq_1s, time_1s, leq_tot_1 = _Leq_1s(arr,fs_cal)
    l_eq_1s.append(l_eq_1s[-1])
    l_eq_125ms.append(l_eq_125ms[-1])

    fig, ax = plt.subplots(figsize=(8,7))
    ax.step(time_125ms, l_eq_125ms, where="post",label=str("$L_{eq,125ms}$ - $L_{eq,tot}$ : "+ str(round(leq_tot_125,1)) +" dB"))
    ax.step(time_1s, l_eq_1s,where="post", label=str("$L_{eq,1s}$ - $L_{eq,tot}$: " + str(round(leq_tot_125,1)) +" dB"))
    ax.legend()
    ax.set_title("Equivalent sound pressure level for \n fast (125 ms) and slow (1s) time constants")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnitude [dB]")
    plt.grid()
    plt.show()

def _Lp_from_third_oct(arr):
    Lp = 0
    for i in arr: Lp += 10**(i/10)
    return 10*np.log10(Lp)

def _sort_into_third_octave_bands(freq, array, third_oct):
    third_octave_banks = []
    single_bank = []
    i = 0

    for it, x in enumerate(freq):
        if (x > third_oct[-1]) : break
        if (x >= third_oct[i] and x < third_oct[i+1]): 
            single_bank.append(round(array[it],2))
        if (x >= third_oct[i+1]):
            third_octave_banks.append(single_bank)
            i += 1
            single_bank = []
            single_bank.append(round(array[it],2))
    filtered_array = []
    for n in third_octave_banks : filtered_array.append(_Lp_from_third_oct(n))
    return filtered_array

def _plot_fft_and_third_oct(freq, fft, third_oct_val, third_oct_step):
    fig, ax = plt.subplots(figsize=(8,7))
    ax.plot(time_125ms, l_eq_125ms, where="post",label=str("$L_{eq,125ms}$ - $L_{eq,tot}$ : "+ str(round(leq_tot_125,1)) +" dB"))
    ax.step(time_1s, l_eq_1s,where="post", label=str("$L_{eq,1s}$ - $L_{eq,tot}$: " + str(round(leq_tot_125,1)) +" dB"))
    ax.legend()
    ax.set_title("FFT- and Third octave bank of sound signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(values.x_ticks_third_octave[values.third_octave_start:])
    ax.set_xtickslabels(values.x_ticks_third_octave_labels[values.third_octave_start])
    ax.legend()
    fig.savefig("FFT&Third_oct.png")

    plt.show()
    


if __name__ == '__main__':
    
    fs_sound_file, sound_file = wavfile.read("A0_lydfil.wav")
    fs_cal, cal_before = wavfile.read("cal_before.wav")
    fs_cal, cal_after = wavfile.read("cal_after.wav")

    
    #print("A:", _scaling_factor(cal_before))

    _check_calibration(cal_before,cal_after)
    array = _scale_array(sound_file,cal_before)
    start = values.start*fs_cal
    stop = values.stop*fs_cal
    print("********\n Sampling frequency: {0} Hz -- length of file {1} minutes\n Start point: {2} s -- End point: {3} s \n ***********".format(fs_cal, len(array)/fs_cal, values.start, values.stop))

    array = array[start:stop]
    
    freq, fft = _fft_signal(array, fs_sound_file)
    print("Lp signal:",_calculate_Lp(array))

    filtered_signal = _sort_into_third_octave_bands(freq, fft, values.third_octave_lower[4:])
    
    plt.step(values.third_octave_center_frequencies[4:], filtered_signal, color="blue", label="FFT for sound signal with NFFT={} s".format(values.stop - values.start))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xscale("log")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":")
    plt.xscale("log")
    #plt.xlim(32.5, 12000)
    plt.legend()
    plt.show()
    
    l_eq_125ms, time_125ms, leq_tot_125 = _Leq_125ms(array,fs_cal)
    l_eq_1s, time_1s, leq_tot_1 = _Leq_1s(array,fs_cal)
    

    _sort_into_third_octave_bands(values.third_octave_center_frequencies)



    


    #_plot_step_Leq(array, fs_cal)



  

    #print(_calc_p_peak(94))
