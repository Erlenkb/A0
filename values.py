p0 = 20*10**(-6)  # p refvalue
cal_value = 94    # Calibration value in dB
start = 40       # Start of the sound signal in seconds
stop = 46         # End of the sound signal in seconds
third_octave_start = 4   # Start of third octave band array -- 4 means starting from 31.5 Hz
freq_min = 30     # Xlim minimum value for the frequency plot
freq_max = 10000  # Xlim maximum value for the frequency plot

nfft = 48000  # Set to zero if nfft value should be length of array
A_weight_freq = True # If set to False, A weight will be found straight from the A-weight filter on the sound pressure signal





#######  Global array values that shall not be changed #########
x_ticks_third_octave = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
x_ticks_third_octave_labels = ["50","100", "200", "500", "1k", "2k", "5k", "10k"]

THIRD_OCTAVE_A_WEIGHTING = [
    -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2,
    -1.9, -0.8, +0.0, +0.6, +1.0, +1.2, +1.3, +1.2, +1.0, +0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
]

THIRD_OCTAVE_A_WEIGHTING = [
    -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2,
    -1.9, -0.8, +0.0, +0.6, +1.0, +1.2, +1.3, +1.2, +1.0, +0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
]

third_octave_center_frequencies = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000]
third_octave_lower = [11.2, 14.1, 17.8, 22.4, 28.2, 35.5, 44.7, 56.2, 70.8, 89.1, 112, 141, 178, 224, 282, 355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467, 5623, 7079, 8913, 11220, 14130, 17780,22390]


