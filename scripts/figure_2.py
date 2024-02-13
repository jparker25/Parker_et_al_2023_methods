"""
Generate Figure 2.

Usage: python figure_2.py
"""

import numpy as np
from matplotlib import pyplot as plt
import os, sys
from scipy.fft import fft, fftfreq

# Change to the absolute path to the STReaC toolbox
sys.path.append("/path/to/streac")
import poisson_spike_train as poisson
import excitation_check as exch
import inhibition_check as inch
from helpers import *

# Set random seed
np.random.seed(24)

# Set parameters
rate = 25
t0 = 0
tf = 10
sigma = 25
mu = 25

# Generate spike train
spikes = poisson.spike_train_generator(rate, tf, t_start=0)

# Generate SDFs and ISIFs at different bandwidths and mus
time = np.linspace(0, tf, tf * 1000)
sdf = exch.kernel(spikes, time, bandwidth=sigma / 1000)
sdf75 = exch.kernel(spikes, time, bandwidth=sigma * 3 / 1000)
sdf250 = exch.kernel(spikes, time, bandwidth=sigma * 10 / 1000)
isif = inch.isi_function(spikes, time, avg=mu)
isif250 = inch.isi_function(spikes, time, avg=mu * 10)

# Create figure
fig = plt.figure(dpi=300, figsize=(12, 4), tight_layout=True)
gs = fig.add_gridspec(2, 4)

axes = [
    fig.add_subplot(gs[:, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[:, 2]),
    fig.add_subplot(gs[:, 3]),
]

# Plot SDF and SDF at larger bandwidth with spikes
axes[0].plot(time, sdf, color="blue", label="$\sigma=25$ms")
axes[0].plot(time, sdf250, color="red", label="$\sigma = 250$ms")
axes[0].scatter(spikes, np.ones(spikes.shape[0]) * -1, marker="|", s=50, color="k")
axes[0].legend(edgecolor="gray", fontsize="xx-small", loc="upper right")
axes[0].set_ylabel("SDF (sps)")
axes[0].set_xlabel("Time (s)")

# Find and plot power spectrums for SDFs
pxx = (2 / sdf.shape[0]) * np.abs(fft(sdf - np.mean(sdf))[1 : sdf.shape[0] // 2]) ** 2
pxx250 = (2 / sdf250.shape[0]) * np.abs(
    fft(sdf250 - np.mean(sdf250))[1 : sdf250.shape[0] // 2]
) ** 2
freqs = fftfreq(sdf.shape[0], 1e-3)[1 : sdf.shape[0] // 2]
axes[1].semilogy(freqs, pxx, color="blue", label="$\sigma=25$ms")
axes[1].semilogy(freqs, pxx250, color="red", label="$\sigma=250$ms")
axes[1].semilogy(freqs, pxx / pxx250, color="gray", label="Ratio")
axes[1].legend(edgecolor="gray", fontsize="xx-small")
axes[1].set_xlim([freqs[1], 50])
axes[1].set_ylim([10e-3, 10e6])
axes[1].set_ylabel("Power")

# Find and plot power spectrums for ISIFs
pxx = (2 / isif.shape[0]) * np.abs(
    fft(isif - np.mean(isif))[1 : isif.shape[0] // 2]
) ** 2
pxx250 = (2 / isif250.shape[0]) * np.abs(
    fft(isif250 - np.mean(isif250))[1 : isif250.shape[0] // 2]
) ** 2
freqs = fftfreq(isif.shape[0], 1e-3)[1 : isif.shape[0] // 2]
axes[2].semilogy(freqs, pxx, color="red", label="$\mu=25$ms")
axes[2].semilogy(freqs, pxx250, color="blue", label="$\mu=250$ms")
axes[2].semilogy(freqs, pxx250 / pxx, color="gray", label="Ratio")
axes[2].set_xlim([freqs[1], 50])
axes[2].set_ylabel("Power")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].legend(edgecolor="gray", fontsize="xx-small")

# Plot ISIF at different mus
axes[3].plot(time, isif, color="red", label="$\mu=25$ms")
axes[3].plot(time, isif250, color="blue", label="$\mu=250$ms")
axes[3].scatter(spikes, np.zeros(spikes.shape[0]) * -1, marker="|", s=50, color="k")
axes[3].legend(edgecolor="gray", fontsize="xx-small")
axes[3].set_ylabel("ISIF (s)")
axes[3].set_xlabel("Time (s)")

# Plot 1/sdf and ISIF for comparison
axes[4].plot(time, 1 / sdf250, color="red", label="1/SDF, $\sigma=250$ms")
axes[4].plot(time, isif250, color="blue", label="ISIF, $\mu=250$ms", alpha=0.75)
axes[4].scatter(spikes, np.zeros(spikes.shape[0]) * -1, marker="|", s=50, color="k")
axes[4].legend(edgecolor="gray", fontsize="xx-small")
axes[4].set_xlabel("Time (s)")
axes[4].set_ylim([axes[3].get_ylim()[0], axes[3].get_ylim()[1]])

# Figure clean up
labels = ["A", "B", "C", "D", "E"]
for i in range(len(axes)):
    axes[i].text(
        0.03,
        0.98,
        labels[i],
        fontsize=16,
        transform=axes[i].transAxes,
        fontweight="bold",
        color="gray",
    )

makeNice(axes)
plt.savefig("../figures/figure_2.pdf")
plt.close()

os.system("open ../figures/figure_2.pdf")
