from tkinter import N
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as signal
import values

def _time_to_discrete(timeval,fs):
    return int(timeval * fs)

def _split_padded(a,n):
     padding = (-len(a))%n
     return np.split(np.concatenate((a,np.zeros(padding))),n)

def _Leq_125ms(array,fs):
    N = _time_to_discrete(0.125, fs)
    it = len(array) // N
    
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

def _Lp_from_freq(fft, freq):
    it = []
    for i, val in enumerate(freq):
        if (val >= 0 and val <= 22390):
            it.append(i)
    Lp = 0
    for i in it:
        Lp += 10**(fft[i]/10)
    return 10*np.log10(Lp)

def _fft_signal(array, fs):
    N = len(array)
    y = np.fft.fft(array)[0:int(N/2)]/N
    y = 1.41*y
    p_y = 20*np.log10(np.abs(y) / values.p0)
    f = fs*np.arange((N/2))/N
    return f, p_y

def _A_weight(array, fs):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = -2

    NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                   [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    DENs = np.polymul(np.polymul(DENs, [1, 2 * np.pi * f3]),
                   [1, 2 * np.pi * f2])

    b, a = signal.bilinear(NUMs, DENs, fs)
    y = signal.lfilter(b,a,array)
    return y

def _calculate_Lp(array):
    rms_pressure = np.sqrt(np.mean(array**2))
    return 20*np.log10(rms_pressure / values.p0)

def _calc_p_peak(cal_value):
    p_rms = values.p0 * 10**(cal_value/20)
    p_peak = p_rms / 0.707
    #print("Pressure peak: {}".format(p_peak))
    return round(p_peak,9)

def _scaling_factor(cal_array):
    p_peak = _calc_p_peak(values.cal_value)
    d_peak = np.mean(np.sort(cal_array[len(cal_array) // 4 : len(cal_array)-len(cal_array) // 4])[-10:])
    #print("Discrete peak: {}".format(d_peak))
    return d_peak / p_peak

def _scale_array(array, cal_array):
    A = _scaling_factor(cal_array)    
    return array / A, A

def _check_calibration(cal_before, cal_after):
    cal_before, A = _scale_array(cal_before, cal_before)
    cal_after, A = _scale_array(cal_after, cal_after)

    Lp_before = _calculate_Lp(cal_before)
    Lp_after = _calculate_Lp(cal_after)
    return Lp_before, Lp_after, A

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
    fig.savefig("Leq,fast,slow_{0}s_to_{1}s.png".format(values.start,values.stop))
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

def _plot_fft_and_third_oct(freq, fft, third_oct_val, third_oct_step, fft_A, freq_A, third_oct_val_A):
    """ Takes in FFT and third octave filtered signal for both Z- and A-weighted signal and plot the values in one figure

    Args:
        freq (_type_): Frequency for the Z-weighted signal
        fft (_type_): FFT of the Z-weghted signal
        third_oct_val (_type_): Y values for the Z-weighted signal filtered into third octave bands
        third_oct_step (_type_): Step values for the thid octave values
        fft_A (_type_): FFT of the A-weighted signal
        freq_A (_type_): Frequency values for the A-weighted signal
        third_oct_val_A (_type_): Values for the A-weighted signal filtered into third octave bands

    Returns:
        _type_: Pressure level for FFT and third octave bands for both Z- and A-weighted levels
    """
    
    fig, ax = plt.subplots(figsize=(8,7))

    Lp_fft = _Lp_from_freq(fft,freq)
    Lp_third_oct = _Lp_from_freq(third_oct_val,third_oct_step)
    Lp_fft_A = _Lp_from_freq(fft_A,freq_A)
    Lp_third_oct_A = _Lp_from_freq(third_oct_val_A,third_oct_step)

    ax.plot(freq, np.abs(fft), color="darkorange",label="FFT Z-weighted")
    ax.plot(freq_A, np.abs(fft_A), color="olive",label="FFT A-weighted")
    ax.step(third_oct_step, third_oct_val, linewidth=2.2, where="mid", 
            color="dimgray", label="1/3-octave filter Z-weighted")
    ax.step(third_oct_step, third_oct_val_A, linewidth=2.2, 
            color="forestgreen", where="mid", label="1/3-octave filter A-weighted")
    
    ax.set_title("FFT- and Third octave bank of Z- and A-weighted sound signal") #\nLp,fft: {0} dB - Lp,fft,A-weight: {2} dB\nLp,Third_oct: {1} dB - Lp,Third_oct,A-weight: {3} dB".format(round(Lp_fft,1),round(Lp_third_oct,1),round(Lp_fft_A,1),round(Lp_third_oct_A,1)))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(values.x_ticks_third_octave)
    ax.set_xticklabels(values.x_ticks_third_octave_labels)
    ax.set_xlim(30,5000)
    ax.legend()
    fig.savefig("FFT&Third_oct_{0} s_to_{1} s.png".format(values.start,values.stop))

    plt.show()
    return Lp_fft, Lp_fft_A, Lp_third_oct, Lp_third_oct_A

def _print_stuff(fs_sound_file=0, sound_file=0, cal_before=0, cal_after=0, A=0, Lp_fft=0, Lp_fft_A=0, third_oct=0, third_oct_A=0, cal_before_Lp=0,cal_after_Lp=0, array=0):
    print("********** -- Important values -- **********\n")
    
    print("Sampling frequency: {0} Hz -- length of file {1} Seconds \nStart point: {2} s -- End point: {3} s"
          .format(fs_cal, len(array)/fs_cal, values.start, values.stop))
    
    print("Scaling factor A={}".format(round(A,1)))
    
    print("Calibration before measurements: {0:.1f} dB\nCalibration after measurements: {1:.1f} dB "
          .format(cal_before_Lp, cal_after_Lp))
    
    print("Total Lp from entire sound signal Z-weighted: {0:.1f} dB \nTotal Lp from Third octave bands Z-weighted: {1:.1f}".format(_calculate_Lp(array), Lp_fft))
    
    print("\t\t Z-weighted \t A-weighted\n Lp,fft\t\t{0:.1f} dB\t\t{1:.1f} dB\n Lp,1/3 \t{2:.1f} dB \t{3:.1f} dB".format(Lp_fft, Lp_fft_A, third_oct, third_oct_A))
    
    print("\n******************************************")


if __name__ == '__main__':
    
    fs_sound_file, sound_file = wavfile.read("A0_lydfil.wav")
    fs_cal, cal_before = wavfile.read("cal_before.wav")
    fs_cal, cal_after = wavfile.read("cal_after.wav")
    
    cal_before_Lp, cal_after_Lp, A = _check_calibration(cal_before,cal_after)
    array, A = _scale_array(sound_file,cal_before)
    start = values.start*fs_cal
    stop = values.stop*fs_cal

    array = array[start:stop]
    array_A_weighted = _A_weight(array, fs_sound_file)
    
    l_eq_125ms, time_125ms, leq_tot_125 = _Leq_125ms(array,fs_cal)
    l_eq_1s, time_1s, leq_tot_1 = _Leq_1s(array,fs_cal)
    
    freq, fft = _fft_signal(array, fs_sound_file)
    freq_A, fft_A = _fft_signal(array_A_weighted, fs_sound_file)

    filtered_signal = _sort_into_third_octave_bands(freq, fft, 
                                                    values.third_octave_lower[values.third_octave_start:])
    filtered_signal_A = _sort_into_third_octave_bands(freq_A, fft_A, 
                                                      values.third_octave_lower[values.third_octave_start:])


    Lp_fft, Lp_fft_A, third_oct, third_oct_A  = _plot_fft_and_third_oct(freq, fft, filtered_signal, 
                            values.third_octave_center_frequencies[values.third_octave_start:], 
                            fft_A, freq_A, filtered_signal_A)

    
    _plot_step_Leq(array, fs_cal)
    
    _print_stuff(fs_sound_file, sound_file, cal_before, cal_after, A, Lp_fft, Lp_fft_A, third_oct, third_oct_A, cal_before_Lp,cal_after_Lp, array)
    
    

  


    


    



  

    #print(_calc_p_peak(94))
