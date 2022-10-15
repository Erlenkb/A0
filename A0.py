from tkinter import N
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as signal
import values

####### Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################



def _time_to_discrete(timeval,fs):
    """Returns discrete timevalue
    Args:
        timeval (_type_): Takes in time interval in s
        fs (_type_): Samplingfrequency

    Returns:
        _type_: Discrete time value
    """
    return int(timeval * fs)

def _split_padded(a,n):
    """Padd end of array to make room for even splitting in _Leq functions

    Args:
        a (_type_): array
        n (_type_): split length

    Returns:
        _type_: Array divided into even slices.
    """
    padding = (-len(a))%n
    return np.split(np.concatenate((a,np.zeros(padding))),n)

def _Leq_125ms(array,fs):
    """Create L_eq with fast (125 ms) time interval

    Args:
        array (_type_): Sound signal array
        fs (_type_): Samplingfrequency

    Returns:
        _type_: array containing Leq and the timestep values
    """
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
    """Create L_eq with slow (1 s) time interval

    Args:
        array (_type_): Sound signal array
        fs (_type_): Samplingfrequency

    Returns:
        _type_: array containing Leq and the timestep values
    """
    N = _time_to_discrete(1, fs)
    it = len(array) // N
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
    """Generate Lp from array in frequency domain

    Args:
        fft (_type_): FFT array
        freq (_type_): Frequency array

    Returns:
        _type_: _description_
    """
    it = []
    for i, val in enumerate(freq):
        if (val >= 0 and val <= 22390):
            it.append(i)
    Lp = 0
    for i in it:
        Lp += 10**(fft[i]/10)
    return 10*np.log10(Lp)

def _runningMeanFast(x, N):
    """
    :param x: Signal array
    :param N: Window_length
    :return: array with running mean values
    """
    return 10*np.log10(np.convolve(10**(x/10), np.ones((N,))/N)[(N-1):])

def _fft_signal(array, fs):
    """Generate the FFT and frequency axis values of the time signal

    Args:
        array (_type_): Sound pressure array
        fs (_type_): Samplingfrequency

    Returns:
        _type_: Frequency array and FFT values
    """
    if (values.nfft != 0) : N = values.nfft
    if (values.nfft == 0) : N = len(array)
    y = np.fft.fft(array, N)[0:int(N/2)]/N
    #y = _runningMeanFast(y,2)
    #y = np.sqrt(2)*y
    p_y = 20*np.log10(np.abs(y) / values.p0)
    f = N*np.arange((N/2))/N
    return f, p_y

def _A_weight(array, fs):
    """Calculate the A weighted filter ad applying it to the sound pressure array.
    Constants are given in the IEC

    Args:
        array (_type_): Sound pressure array
        fs (_type_): Samplingfrequency

    Returns:
        _type_: The A-weighted sound pressure array
    """
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
    """Calculate the sound pressure level for a sound pressure array

    Args:
        array (_type_): Sound pressure array

    Returns:
        _type_: Lp
    """
    rms_pressure = np.sqrt(np.mean(array**2))
    return 20*np.log10(rms_pressure / values.p0)

def _calc_p_peak(cal_value):
    """Find the pressure peak for the calibration signal

    Args:
        cal_value (_type_): Calibration array

    Returns:
        _type_: Pressure peak
    """
    p_rms = values.p0 * 10**(cal_value/20)
    p_peak = p_rms / 0.707
    return round(p_peak,9)

def _scaling_factor(cal_array):
    """Find the scaling factor to apply to the sound pressure array which are from a wav signal containing values between 0-1

    Args:
        cal_array (_type_): Calibration array

    Returns:
        _type_: Scaling factor A
    """
    p_peak = _calc_p_peak(values.cal_value)
    d_peak = np.mean(np.sort(cal_array[len(cal_array) // 4 : len(cal_array)-len(cal_array) // 4])[-10:])
    #print("Discrete peak: {}".format(d_peak))
    return d_peak / p_peak

def _scale_array(array, cal_array):
    """Apply the scaling factor to a sound pressure array

    Args:
        array (_type_): Sound pressure array to be scaled
        cal_array (_type_): Sound pressure array from calibration signal

    Returns:
        _type_: Scaled array and A factor
    """
    A = _scaling_factor(cal_array)    
    return array / A, A

def _check_calibration(cal_before, cal_after):
    """Check calibration signal before and after measurement

    Args:
        cal_before (_type_): Calibration signal before measurement
        cal_after (_type_): Calibration signal after measurement

    Returns:
        _type_: Sound pressure level, Lp, before and after, as well as scaling factor A
    """
    cal_before, A = _scale_array(cal_before, cal_before)
    cal_after, A = _scale_array(cal_after, cal_after)

    Lp_before = _calculate_Lp(cal_before)
    Lp_after = _calculate_Lp(cal_after)
    return Lp_before, Lp_after, A

def _plot_step_Leq(arr, fs):
    """Plot the Leq values for the given time interval

    Args:
        arr (_type_): Sound pressure array
        fs (_type_): Samplingfrequency
    """
    l_eq_125ms, time_125ms, leq_tot_125 = _Leq_125ms(arr,fs_cal)
    l_eq_1s, time_1s, leq_tot_1 = _Leq_1s(arr,fs_cal)
    l_eq_1s.append(l_eq_1s[-1])
    l_eq_125ms.append(l_eq_125ms[-1])

    fig, ax = plt.subplots(figsize=(8,7))
    ax.step(time_125ms, l_eq_125ms, color="olive", where="post",label=str("$L_{eq,125ms}$ - $L_{eq,tot}$ : "+ str(round(leq_tot_125,1)) +" dB"))
    ax.step(time_1s, l_eq_1s,where="post", color="forestgreen", label=str("$L_{eq,1s}$ - $L_{eq,tot}$: " + str(round(leq_tot_125,1)) +" dB"))
    ax.legend()
    ax.set_title("Equivalent sound pressure level for \n fast (125 ms) and slow (1s) time constants")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnitude [dB]")
    plt.grid()
    fig.savefig("Pictures\Leq,fast,slow_{0}s_to_{1}s.png".format(values.start,values.stop))
    plt.show()

def _Lp_from_third_oct(arr):
    """Return the sound pressure level from the third octave band array

    Args:
        arr (_type_): Third octave band array

    Returns:
        _type_: Lp
    """
    Lp = 0
    for i in arr: Lp += 10**(i/10)
    return 10*np.log10(Lp)

def _sort_into_third_octave_bands(freq, array, third_oct, A_weight_val):
    """Takes in the FFT and frequency values and sort it into third octave bands

    Args:
        freq (_type_): Frequency array
        array (_type_): FFT aray
        third_oct (_type_): Third octave lower frequency values

    Returns:
        _type_: Filtered array using third octave bands
    """
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
    filtered_signal_A = []
    for n in third_octave_banks : filtered_array.append(_Lp_from_third_oct(n))
    for n, val in enumerate(filtered_array) : filtered_signal_A.append(val + A_weight_val[n])
    return filtered_array, filtered_signal_A

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

    ax.plot(freq, np.abs(fft), color="olive",label="FFT Z-weighted")
    #ax.plot(freq_A, np.abs(fft_A), color="olive",label="FFT A-weighted")
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
    ax.set_xlim(values.freq_min, values.freq_max)
    ax.legend()
    fig.savefig("Pictures\FFT&Third_oct_{0} s_to_{1} s.png".format(values.start,values.stop))
    
    plt.show()
    return Lp_fft, Lp_fft_A, Lp_third_oct, Lp_third_oct_A

def _print_stuff(fs_sound_file=0, sound_file=0, cal_before=0, cal_after=0, A=0, Lp_fft=0, Lp_fft_A=0, third_oct=0, third_oct_A=0, cal_before_Lp=0,cal_after_Lp=0, array=0):
    """Print important values in a good way :)

    Args:
        fs_sound_file (int, optional): _description_. Defaults to 0.
        sound_file (int, optional): _description_. Defaults to 0.
        cal_before (int, optional): _description_. Defaults to 0.
        cal_after (int, optional): _description_. Defaults to 0.
        A (int, optional): _description_. Defaults to 0.
        Lp_fft (int, optional): _description_. Defaults to 0.
        Lp_fft_A (int, optional): _description_. Defaults to 0.
        third_oct (int, optional): _description_. Defaults to 0.
        third_oct_A (int, optional): _description_. Defaults to 0.
        cal_before_Lp (int, optional): _description_. Defaults to 0.
        cal_after_Lp (int, optional): _description_. Defaults to 0.
        array (int, optional): _description_. Defaults to 0.
    """
    print("********** -- Important values -- **********\n")
    
    print("- Sampling frequency: {0} Hz -- length of file {1} Seconds \n- Start point: {2} s \n- End point: {3} s"
          .format(fs_cal, len(array)/fs_cal, values.start, values.stop))
    print("- Scaling factor A = {}".format(round(A,1)))
    print("- Calibration before measurements: {0:.1f} dB\n- Calibration after measurements: {1:.1f} dB "
          .format(cal_before_Lp, cal_after_Lp))
    print("- Total Lp from sound pressure signal Z-weighted: {0:.1f} dB \n- Total Lp from Third octave bands Z-weighted: {1:.1f} dB".format(_calculate_Lp(array), Lp_fft))
    print("\t\t Z-weighted \t A-weighted\n Lp,fft\t\t{0:.1f} dB\t\t{1:.1f} dB\n Lp,1/3 \t{2:.1f} dB \t{3:.1f} dB".format(Lp_fft, Lp_fft_A, third_oct, third_oct_A))
    
    print("\n******************************************")



if __name__ == '__main__':
    
    #**********  Generate sound pressure arrays *******
    fs_sound_file, sound_file = wavfile.read("A0_lydfil.wav")
    fs_cal, cal_before = wavfile.read("cal_before.wav")
    fs_cal, cal_after = wavfile.read("cal_after.wav")
    
    #*********   Check calibration, find scaling factor and scale sound pressure array******
    cal_before_Lp, cal_after_Lp, A = _check_calibration(cal_before,cal_after)
    array, A = _scale_array(sound_file,cal_before)
    
    start = values.start*fs_cal
    stop = values.stop*fs_cal
    array = array[start:stop]
    
    #*********   Filter sound pressure array into A-weighted array *******
    array_A_weighted = _A_weight(array, fs_sound_file)
    
    #*********   Calculate the Leq arrays for fast (125 ms) and slow (1 s) time constants  ******
    l_eq_125ms, time_125ms, leq_tot_125 = _Leq_125ms(array,fs_cal)
    l_eq_1s, time_1s, leq_tot_1 = _Leq_1s(array,fs_cal)
    
    #*********   Generate the FFT and frequency array for Z- and A-weighted sound pressure array ******
    freq, fft = _fft_signal(array, fs_sound_file)
    freq_A, fft_A = _fft_signal(array_A_weighted, fs_sound_file)

    #*********   Sort FFT values into third octave bands for Z- and A-weighted FFT signal ******
    filtered_signal, filtered_signal_A = _sort_into_third_octave_bands(freq, fft, 
                                                    values.third_octave_lower[values.third_octave_start:], 
                                                    values.THIRD_OCTAVE_A_WEIGHTING[values.third_octave_start:])
    
    if (values.A_weight_freq == False):
        filtered_signal_A = _sort_into_third_octave_bands(freq_A, fft_A, 
                                                          values.third_octave_lower[values.third_octave_start:])
    
    #*********   Plot the FFT and 1/3 octave band values and return the Lp values for these *****
    Lp_fft, Lp_fft_A, third_oct, third_oct_A  = _plot_fft_and_third_oct(freq, fft, filtered_signal, 
                            values.third_octave_center_frequencies[values.third_octave_start:], 
                            fft_A, freq_A, filtered_signal_A)

    #*********   plot the Leq values in steps   ******
    _plot_step_Leq(array, fs_cal)
    
    #*********   Print the important parameters ******
    _print_stuff(fs_sound_file, sound_file, cal_before, cal_after, A, 
                 Lp_fft, Lp_fft_A, third_oct, third_oct_A, cal_before_Lp,
                 cal_after_Lp, array)