import numpy as np

import values


array = [1,2,2,1,34,1,3,2,34,1,2,2,21,3,2]


def _nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def _octave_filter():



    return 1


def _calculate_Lp(array):
    avg_pressure = np.average(array)
    return 20*np.log10(avg_pressure / values.p0)



print(_nextpow2(8))

print(_calculate_Lp(array))