# Simulacao do conteudo espectral de raios
# Lazaro Camargo - INPE

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt


b_s = 1e6
a_s = 1e9

t = np.linspace(-1e-9,10e-9,200)

L = len(t)
x = np.zeros(L)

t0 = 0; a = 1




exp_array = np.frompyfunc(mp.exp, 1, 1)


# vv = (b_s*exp_array(t))/(1+exp_array((a_s+b_s)*t))

def f(t2):
    return (b_s*exp_array(t2))/(1+exp_array((a_s+b_s)*t2)) if  t2 > 0 else 0


k = 0
for tx in t:
    x[k] = f(tx)
    k = k+1

plt.figure(1)
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("f(t)")

plt.show()

z = np.array(x.tolist(), dtype=float)

gg = np.fft.fft(z)

#plt.plot(np.abs(gg))
#plt.show()

Y = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), t[1] - t[0])

plt.figure(2)
plt.loglog((2/200)*freq[0:100], np.abs(Y[0:100]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.show()

