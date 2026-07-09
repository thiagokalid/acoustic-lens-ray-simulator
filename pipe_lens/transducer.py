import numpy as np

from scipy.signal import gausspulse

__all__ = ["Transducer"]

class Transducer:
    """
    Classe que contém informações do transdutor usado no ensaio.

    Parameters
    ----------
    pitch : float
        Distância entre o centro de dois elementos subsequentes do transdutor em metros.
    num_elem : int
        Número de elementos do transdutor.
    element_gap : float
        Tamanho do vão entre dois elementos subsequentes em metros.
    fc : float
        Frequência central do transdutor em Hz.
    bw : float
        Banda do transdutor
    bwr : float
        Amplitude de corte da banda do transdutor.
    pulse_type : str
        Tipo da resposta ao impulso do cristal.
    center : tuple[float, float]
        Coordenadas (x, z) do centro do transdutor, em metros.
    """
    def __init__(self, pitch: float = .4e-3, num_elem: int = 64, element_gap:float = .1e-3, fc: float = 5e6, bw: float = .4, bwr: float = -6,
                 pulse_type: str = "gaussian", center: tuple[float, float] = (0., 0.), rotation_ang: float = 0.):
        self.pitch = pitch
        self.num_elem = num_elem
        self.fc = fc
        self.wc = 2 * np.pi * fc
        self.bw = bw  # Hz
        self.bwr = bwr  # dB
        self.pulse_type = pulse_type
        self.xt = np.arange(0, self.num_elem) * pitch
        self.xt -= np.mean(self.xt)
        self.zt = np.zeros_like(self.xt)
        self.elements = np.arange(1, self.num_elem + 1, 1)
        self.element_gap = element_gap
        self.element_width = self.pitch - self.element_gap
        self.center = center
        self.rotation_ang = rotation_ang
        M = np.array([
            [np.cos(rotation_ang), np.sin(rotation_ang)],
            [-np.sin(rotation_ang), np.cos(rotation_ang)]
        ])
        r = np.vstack([self.xt, self.zt])
        r_new = M @ r
        self.xt = r_new[0, :]
        self.zt = r_new[1, :]
        self.xt += self.center[0]
        self.zt += self.center[1]

    def get_coords(self, i: int = -1):
        if i == -1:
            return self.xt, self.zt
        else:
            return self.xt[i], self.zt[i]

    def compute_time_domain_response(self, tspan):
        if self.pulse_type == "gaussian":
            return gausspulse(tspan, fc=self.fc, bw=self.bw, bwr=self.bwr)
        else:
            raise NotImplementedError

    def compute_freq_domain_response(self, omega):
        if self.pulse_type == "gaussian":
            omega_c = 2 * np.pi * self.fc

            # SciPy definition of 'a'
            ref = 10 ** (self.bwr / 20)
            a = - (np.pi * self.fc * self.bw) ** 2 / np.log(ref)

            prefactor = 0.5 * np.sqrt(np.pi / a)

            spectrum = prefactor * (
                    np.exp(-(omega - omega_c) ** 2 / (4 * a)) +
                    np.exp(-(omega + omega_c) ** 2 / (4 * a))
            )

            return spectrum.astype(np.complex64)
        else:
            raise NotImplementedError

    def flip_xcoords(self):
        self.xt = self.xt[::-1]