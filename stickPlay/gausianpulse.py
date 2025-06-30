import numpy as np
import matplotlib.pyplot as plt

def gausian_pulse(k, sigma, tau, mag):
    return mag*np.exp(-(k-tau)**2 / (2 * sigma**2))



k = np.linspace(0, 1, 200)
ko = gausian_pulse(k, 0.02, 55/200, 1)
ti = gausian_pulse(k, 0.01, 100/200, 0.5)
ki = gausian_pulse(k, 0.01, 120/200, 0.6)
te = gausian_pulse(k, 0.01, 150/200, 0.9)

kottikite = ko + ti + ki + te

# プロット
plt.figure(figsize=(8, 4))
plt.plot(k, kottikite, 'b-', linewidth=2)
plt.title('ガウシアンパルス')
plt.xlabel('時間')
plt.ylabel('振幅')
plt.grid(True, alpha=0.3)
plt.show()