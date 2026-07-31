import numpy as np
from pipe_lens.transducer import Transducer
import matplotlib.pyplot as plt

transducer = Transducer()

fs = 125e6

tspan = np.arange(0, 90e-6, 1/fs)

ascan = transducer.compute_time_domain_response

plt.plot(tspan, ascan)