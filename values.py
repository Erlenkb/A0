p0 = 20*10**(-6)
cal_value = 94

import numpy as np
import matplotlib.pyplot as plt
"""
i = 1
x = np.linspace(0,i,i*44100)
x1 = np.linspace(0,i*10, i*10*44100)
y = np.sin(2*np.pi*x*3)
y1 = np.sin(2*np.pi*x1*3)

fft = np.fft.fft(y)
fft1 = np.fft.fft(y1,44100)
freq = np.fft.fftfreq(n = len(fft), d=1/44100)
freq1 = np.fft.fftfreq(n=len(fft1), d=(1/44100))

plt.plot(np.fft.fftshift(freq), np.fft.fftshift(fft),color="blue")
plt.plot(np.fft.fftshift(freq1), np.fft.fftshift(fft1),color="red")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.xscale("log")
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.xscale("log")
    
plt.legend()

plt.show()
plt.plot(x,y)
plt.show()
"""