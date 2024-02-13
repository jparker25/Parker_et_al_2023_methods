"""
Generate Figure 3.

Usage: python figure_3.py
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle, os, sys

from helpers import *

# Change to the absolute path to the STReaC toolbox
sys.path.append("/path/to/streac")
import excitation_check as exch
import inhibition_check as inch

# Read in example data
save_direc = "../methods_results"
csv = f"{save_direc}/all_data.csv"
types = [
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
    "biphasic IE",
    "biphasic EI",
    "no effect",
]
df = pd.read_csv(csv)

# Find test case
sample = df[(df["group"] == "Naive_mice_hsyn-ChR2_in_GPe") & (df["cell_num"] == 62)]
cell_dir = sample["cell_dir"].values[0]
neuron = pickle.load(open(f"{cell_dir}/neuron.obj", "rb"))
trial = 9

# Read in trial spike trains
stim_data_train = pickle.load(
    open(
        f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj",
        "rb",
    )
)
baseline_train = pickle.load(
    open(
        f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj",
        "rb",
    )
)

# Set shuffles and trial percentile
shuffles = 10
percentile = 99

# Set seed
np.random.seed(24)

# Analyze inhibition by shuffles on baseline
### INHIBITION ###
results = np.zeros(
    len(stim_data_train.bin_edges) - 1
)  # Create empty array that will store the classifications of each bin
bl_isi = np.diff(baseline_train.spikes)  # Find the baseline ISI values
bl_shuffles = [
    np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)
]  # Shuffle the bl ISI values for shuffles number of times
bl_spikes = [
    [
        (
            baseline_train.spikes[0] + sum(bl_shuff[0:i])
            if i > 0
            else baseline_train.spikes[0]
        )
        for i in range(len(bl_isi))
    ]
    for bl_shuff in bl_shuffles
]  # Recreate the spikes based on the shuffled ISI values
bl_isi_fs = [inch.isi_function(bl_spike, baseline_train.t) for bl_spike in bl_spikes]
bl_isif_areas = (
    []
)  # Empty array that will store the areas of each bin of the shuffled baseline
for i in range(shuffles):  # Loop through all the shuffles
    for bi in range(
        len(baseline_train.bin_edges) - 1
    ):  # In each shuffle, loop throuhg each bin
        st, ed = (
            np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],
            np.where(baseline_train.t < baseline_train.bin_edges[bi + 1])[0][-1],
        )
        bl_isif_areas.append(
            np.trapz(bl_isi_fs[i][st : ed + 1], x=baseline_train.t[st : ed + 1])
        )  # Find the area of the baseline SDF of the bin and append to array

# Analyze excitation by shuffles on baseline
### EXCITATION ###
bl_isi = np.diff(baseline_train.spikes)  # Find the baseline ISI values
bl_shuffles = [
    np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)
]  # Shuffle the bl ISI values for shuffles number of times
bl_spikes = [
    [
        (
            baseline_train.spikes[0] + sum(bl_shuff[0:i])
            if i > 0
            else baseline_train.spikes[0]
        )
        for i in range(len(bl_isi))
    ]
    for bl_shuff in bl_shuffles
]  # Recreate the spikes based on the shuffled ISI values
bl_sdf_areas = (
    []
)  # Empty array that will store the areas of each bin of the shuffled baseline
bl_sdf_fs = [exch.kernel(bl_spike, baseline_train.t) for bl_spike in bl_spikes]
for i in range(shuffles):  # Loop through all the shuffles
    for bi in range(
        len(baseline_train.bin_edges) - 1
    ):  # In each shuffle, loop throuhg each bin
        st, ed = (
            np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],
            np.where(baseline_train.t < baseline_train.bin_edges[bi + 1])[0][-1],
        )
        # bl_sdf_areas.append(np.trapz(baseline_train.sdf[st:ed+1],x=baseline_train.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
        bl_sdf_areas.append(
            np.trapz(bl_sdf_fs[i][st : ed + 1], x=baseline_train.t[st : ed + 1])
        )

# Generate figure
fig = plt.figure(figsize=(8, 4), dpi=300)
gs = fig.add_gridspec(2, 3)

axes = [
    fig.add_subplot(gs[:, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 2]),
]

# Plot baseline shuffles for ISIF, SDF, and spikes
count = 1
for shuff in bl_spikes:
    axes[0].scatter(shuff, np.ones(len(shuff)) * count, marker="|", s=50, color="k")
    axes[1].plot(
        baseline_train.t,
        (bl_sdf_fs[count - 1] / np.max(bl_sdf_fs[count - 1])) + count,
        color="blue",
    )
    axes[2].plot(
        baseline_train.t,
        (bl_isi_fs[count - 1] / np.max(bl_isi_fs[count - 1])) + count,
        color="blue",
    )
    count += 1

for i in range(3):
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Permutation Number")
    axes[i].set_yticks(np.arange(1, 11))
    axes[i].set_yticklabels(np.arange(0, 10), fontsize=8)

# Plot SDF area histogram
axes[3].hist(
    bl_sdf_areas, bins=20, color="gray", edgecolor="black"
)  # Create a histogram of all the baseline SDF areas
axes[3].vlines(
    np.percentile(bl_sdf_areas, percentile),
    axes[3].get_ylim()[0],
    axes[3].get_ylim()[1],
    color="k",
    linestyle="dashed",
)  # Draw a verticle line where the percentile threshold is on the histogram
axes[3].set_xlabel("Baseline SDF Area Bins")
axes[3].set_ylabel("Count")  # Label the histogram axes

# Plot ISIF area histogram
axes[4].hist(
    bl_isif_areas, bins=20, color="gray", edgecolor="black"
)  # Create a histogram of all the baseline SDF areas
axes[4].vlines(
    np.percentile(bl_isif_areas, percentile),
    axes[4].get_ylim()[0],
    axes[4].get_ylim()[1],
    color="k",
    linestyle="dashed",
)  # Draw a verticle line where the percentile threshold is on the histogram
axes[4].set_xlabel("Baseline ISIF Area Bins")
axes[4].set_ylabel("Count")  # Label the histogram axes

# Figure clean up
makeNice(axes)
labels = ["A", "B", "C", "D", "E", "F", "G"]
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

plt.tight_layout()
plt.savefig("../figures/figure_3.pdf")
plt.close()

os.system("open ../figures/figure_3.pdf")
