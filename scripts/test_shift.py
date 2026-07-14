import numpy as np
import matplotlib.pyplot as plt
from pipe_lens.transducer import Transducer

# define a simulation time-frame:
fs = 125e6
time_span = np.arange(0, 60e-6, 1/fs)

# define propagation medium properties
cl = 5900

# create a linear-phased array transducer
transducer = Transducer(fc=2e6)

# define a punctual reflector
xs, zs = 0, 25e-3
delay = np.sqrt((xs - transducer.xt)**2 + (zs - transducer.zt)**2) / cl
delay_law = np.mean(delay) - delay

bscan = transducer.compute_time_domain_response(np.subtract.outer(time_span, delay))

# compute the equivalent frequency-domain signal
dt = 1/fs
N = len(time_span)
f = np.fft.fftfreq(N, d=dt)
omega = 2 * np.pi * f
domega = 2 * np.pi / (N * dt)

shift = np.exp(np.multiply.outer(-1j * omega, delay))

bscan_w = (
    transducer.compute_freq_domain_response(omega[:, np.newaxis])
    * shift
)

# apply delay-law:
bscan_w *= np.exp(np.multiply.outer(-1j * omega, delay_law))

bscan_ = np.real(np.fft.ifft(bscan_w, axis=0))
# plt.plot(bscan_)


# plot result
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# plt.plot(time_span, ascan)
axs[0].imshow(bscan, extent=(0, 64, time_span[-1] * 1e6, time_span[0] * 1e6), aspect='auto', interpolation='None')
axs[1].imshow(bscan_, extent=(0, 64, time_span[-1] * 1e6, time_span[0] * 1e6), aspect='auto', interpolation='None')
plt.show()

plt.figure()
plt.plot(bscan[:, 31] / bscan[:, 31].max(), label="Time-domain Gaussian Pulse")
plt.plot(bscan_[:, 31] / bscan_[:, 31].max(), label="Freq-domain Gaussian Pulse")
plt.legend()
plt.show()
