"""
Generate Figure 5.

Usage: python figure_5.py
"""

import numpy as np
from matplotlib import pyplot as plt
import os, sys, time, string
from datetime import datetime, timedelta
import scipy.optimize as sciop

# Change to the absolute path to the STReaC toolbox
sys.path.append("/path/to/streac")

import poisson_spike_train as poisson
import excitation_check as exch
import inhibition_check as inch
from helpers import *


# Normalize area
def normalize_fcn(x, t):
    area = np.trapz(x, t)
    return x / area


# 1/x function for a best fit
def func(x, a, b):
    return a / x + b


# Find all baseline areas
def find_baseline_areas(
    spikes, bin_edges, tt, isISIF, shuffles=10, norm=False, avg=250
):
    bl_isi = np.diff(spikes)  # Find the baseline ISI values
    bl_shuffles = [
        np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)
    ]  # Shuffle the bl ISI values for shuffles number of tts
    bl_spikes = [
        [
            spikes[0] + sum(bl_shuff[0:i]) if i > 0 else spikes[0]
            for i in range(len(bl_isi))
        ]
        for bl_shuff in bl_shuffles
    ]  # Recreate the spikes based on the shuffled ISI values
    bl_isi_fs = (
        [inch.isi_function(bl_spike, tt, avg=avg) for bl_spike in bl_spikes]
        if isISIF
        else [exch.kernel(bl_spike, tt, bandwidth=25 / 1000) for bl_spike in bl_spikes]
    )
    if norm:
        for i in range(len(bl_isi_fs)):
            bl_isi_fs[i] = normalize_fcn(bl_isi_fs[i], tt)
    bl_isif_areas = (
        []
    )  # Empty array that will store the areas of each bin of the shuffled baseline
    for i in range(shuffles):  # Loop through all the shuffles
        for bi in range(len(bin_edges) - 1):  # In each shuffle, loop throuhg each bin
            st, ed = (
                np.where(tt <= bin_edges[bi])[0][-1],
                np.where(tt < bin_edges[bi + 1])[0][-1],
            )
            bl_isif_areas.append(
                np.trapz(bl_isi_fs[i][st : ed + 1], x=tt[st : ed + 1])
            )  # Find the area of the baseline SDF of the bin and append to array
    return bl_isif_areas


# Find all stim areas
def find_stim_areas(spikes, bin_edges, tt, isISIF, avg=250, bandwidth=25 / 1000):
    stim_areas = []
    stim_isif = (
        inch.isi_function(spikes, tt, avg=avg)
        if isISIF
        else exch.kernel(spikes, tt, bandwidth=bandwidth)
    )
    for i in range(len(bin_edges) - 1):
        st, ed = (
            np.where(tt <= bin_edges[i])[0][-1],
            np.where(tt < bin_edges[i + 1])[0][-1],
        )
        stim_areas.append(np.trapz(stim_isif[st : ed + 1], x=tt[st : ed + 1]))
    return np.asarray(stim_areas)


# Calculate amount of areas above percentiles
def isif_sdf_ratio(
    stim_sdf_areas, baseline_sdf_percentile, stim_isif_areas, baseline_isif_percentile
):
    num_sdf = np.count_nonzero(stim_sdf_areas < baseline_sdf_percentile)
    num_isif = np.count_nonzero(stim_isif_areas >= baseline_isif_percentile)
    return num_isif, num_sdf


# Set random seed
np.random.seed(24)

# Set example parameters
high_rate = 40
low_rate = 10
decrease = 50
t0 = 0
tf = 10
bin_width = 0.5
sigma = 25 / 1000
mu = 250

# Make true to generate poisson spike train data (scatter plot panel)
generate_spike_trains = False

# Set up time and bins
tt = np.linspace(0, tf, tf * 1000)
bin_edges = np.arange(0, tf + bin_width, bin_width)

# Generate poisson trains
high_spikes = poisson.spike_train_generator(high_rate, tf)
low_spikes = poisson.spike_train_generator(low_rate, tf)
high_spikes_reduced = poisson.spike_train_generator(
    high_rate * (100 - decrease) / 100, tf
)
low_spikes_reduced = poisson.spike_train_generator(
    low_rate * (100 - decrease) / 100, tf
)

# Generate SDFs and ISIFs
sdf_high = exch.kernel(high_spikes, tt, bandwidth=sigma)
isif_high = inch.isi_function(high_spikes, tt, avg=mu)
sdf_low = exch.kernel(low_spikes, tt, bandwidth=sigma)
isif_low = inch.isi_function(low_spikes, tt, avg=mu)

# Generate reduced SDFs and ISIFs
sdf_high_reduced = exch.kernel(high_spikes_reduced, tt, bandwidth=sigma)
isif_high_reduced = inch.isi_function(high_spikes_reduced, tt, avg=mu)
sdf_low_reduced = exch.kernel(low_spikes_reduced, tt, bandwidth=sigma)
isif_low_reduced = inch.isi_function(low_spikes_reduced, tt, avg=mu)

# Generate figure
fig = plt.figure(figsize=(12, 8), dpi=300, tight_layout=True)
gs = fig.add_gridspec(4, 6)
axes = [
    fig.add_subplot(gs[0, :2]),
    fig.add_subplot(gs[1, :2]),
    fig.add_subplot(gs[2, :2]),
    fig.add_subplot(gs[3, :2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 2:4]),
    fig.add_subplot(gs[2, 2:4]),
    fig.add_subplot(gs[3, 2:4]),
]

spikes_to_plot = [
    high_spikes,
    high_spikes_reduced,
    high_spikes,
    high_spikes_reduced,
    low_spikes,
    low_spikes_reduced,
    low_spikes,
    low_spikes_reduced,
]


params = []

row = 0
# iterate through axes and plot high and low rates, baseline data and corresponding stimulus for comparisons
for k in range(len(axes)):
    if (k % 4) == 0:
        sdf_areas = find_baseline_areas(
            spikes_to_plot[k], bin_edges, tt, False, norm=False
        )
        sdf_percentile = np.percentile(sdf_areas, 99)
        sdf_low_percentile = np.percentile(sdf_areas, 1)
        axes[k].hist(
            sdf_areas,
            bins=20,
            alpha=0.5,
            edgecolor="k",
            color="blue",
            label="Shuffled SDF Areas",
        )
        axes[k].vlines(
            sdf_percentile, 0, axes[k].get_ylim()[1], linestyle="dashed", color="k"
        )
        axes[k].vlines(
            sdf_low_percentile, 0, axes[k].get_ylim()[1], linestyle="dotted", color="k"
        )
        params.append(sdf_low_percentile)

    elif (k % 4) == 1:
        stim_areas = find_stim_areas(
            spikes_to_plot[k], bin_edges, tt, False, avg=mu, bandwidth=sigma
        )
        axes[k].bar(
            list(range(1, len(stim_areas) + 1)),
            stim_areas,
            color="gray",
            edgecolor="black",
            width=1,
            label="Comparison SDF Areas",
        )
        axes[k].hlines(sdf_percentile, 0.5, 20.5, color="k", linestyle="dashed")
        axes[k].hlines(sdf_low_percentile, 0.5, 20.5, color="k", linestyle="dotted")
        axes[k].set_xticks(np.linspace(0, 20, 5))
        axes[k].set_xticklabels(np.linspace(0, 10, 5))
        axes[k].set_xlim([0.5, 20.5])
        params.append(stim_areas)
    elif (k % 4) == 2:
        isif_areas = find_baseline_areas(
            spikes_to_plot[k], bin_edges, tt, True, norm=False
        )
        isif_percentile = np.percentile(isif_areas, 99)
        isif_low_percentile = np.percentile(isif_areas, 1)
        axes[k].hist(
            isif_areas,
            bins=20,
            alpha=0.5,
            edgecolor="k",
            color="blue",
            label="Shuffled ISIF Areas",
        )
        axes[k].vlines(
            [isif_percentile], 0, axes[k].get_ylim()[1], linestyle="dashed", color="k"
        )
        axes[k].vlines(
            [isif_low_percentile],
            0,
            axes[k].get_ylim()[1],
            linestyle="dotted",
            color="k",
        )
        params.append(isif_percentile)

    else:
        stim_areas = find_stim_areas(
            spikes_to_plot[k], bin_edges, tt, True, avg=mu, bandwidth=sigma
        )
        axes[k].bar(
            list(range(1, len(stim_areas) + 1)),
            stim_areas,
            color="gray",
            edgecolor="black",
            width=1,
            label="Comparison ISIF Areas",
        )
        axes[k].hlines(isif_percentile, 0.5, 20.5, color="k", linestyle="dashed")
        axes[k].hlines(isif_low_percentile, 0.5, 20.5, color="k", linestyle="dotted")
        axes[k].set_xticks(np.linspace(0, 20, 5))
        axes[k].set_xticklabels(np.linspace(0, 10, 5))
        axes[k].set_xlim([0.5, 20.5])

# Label plots
axes[0].set_title("High Rate")
axes[4].set_title("Low Rate")
axes[0].set_ylabel("Count")
axes[2].set_ylabel("Count")
axes[1].set_ylabel("SDF Area")
axes[3].set_ylabel("ISIF Area")

# Clean up figure
makeNice(axes)
labels = string.ascii_uppercase
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
    if ((i % 4) == 0) or ((i % 4) == 2):
        axes[i].set_xlabel("Area")
    else:
        axes[i].set_xlabel("Time (s)")


# Adjust parameters to generate new set of poisson trains to compare
tf = 100
t0 = 0
Nrates = 1000
max_rate = 50
min_rate = 5
if generate_spike_trains:
    # Create random distribution of rates
    random_rates = np.sort(
        np.random.randint(low=min_rate, high=max_rate, size=(Nrates))
    )
    # Set parameters - advised to keep the same as beginning of file
    bin_width = 0.5
    sigma = 25 / 1000
    mu = 250
    tt = np.linspace(0, tf, tf * 1000)
    bin_edges = np.arange(0, tf + bin_width, bin_width)
    data = np.zeros((Nrates, 4))
    tot_tt = 0
    tremaining = 0
    sim = 1
    # Iterate through rates and determine stimulus areas inhibted and excited
    for k in range(Nrates):
        start = time.time()
        rate = random_rates[k]
        spikes = poisson.spike_train_generator(rate, tf)
        spikes_reduced = poisson.spike_train_generator(
            rate * (100 - decrease) / 100, tf
        )
        data[k, 0] = spikes.shape[0] / tf
        baseline_sdf_percentile = np.percentile(
            find_baseline_areas(spikes, bin_edges, tt, False, norm=False), 1
        )
        stim_sdf_areas = find_stim_areas(
            spikes_reduced, bin_edges, tt, False, avg=mu, bandwidth=sigma
        )
        baseline_isif_percentile = np.percentile(
            find_baseline_areas(spikes, bin_edges, tt, True, norm=False), 99
        )
        stim_isif_areas = find_stim_areas(
            spikes_reduced, bin_edges, tt, True, avg=mu, bandwidth=sigma
        )
        num_isif, num_sdf = isif_sdf_ratio(
            stim_sdf_areas,
            baseline_sdf_percentile,
            stim_isif_areas,
            baseline_isif_percentile,
        )
        # Store amount that are inhibited and excited
        data[k, 1:3] = [num_isif, num_sdf]
        data[k, 3] = num_isif / num_sdf if num_sdf > 0 else num_isif
        end = time.time()
        diff = end - start
        tot_tt += diff
        tremaining = (tot_tt / (sim)) * (Nrates - sim)
        est_done = datetime.now() + timedelta(seconds=tremaining)
        trem = time.strftime("%H:%M:%S", time.gmtime(tremaining))
        print(
            f"Sim: {sim}/{Nrates}, Time: {diff:.02f}, TRemaining: {trem} Est Done: {est_done.ctime()}"
        )
        sim += 1
    # Save data
    data = data[data[:, 0].argsort()]
    np.savetxt(
        f"data_tf_{tf}_hz_{min_rate}_{max_rate}_N_{Nrates}.txt",
        data,
        newline="\n",
        delimiter="\t",
    )

# Read in poisson data trains
data = np.loadtxt(f"data_tf_{tf}_hz_{min_rate}_{max_rate}_N_{Nrates}.txt")
ratios = data[(data[:, 2] > 0) & (data[:, 1] > 0)]
infs = data[(data[:, 2] == 0) & (data[:, 1] > 0)]
sdfs = data[(data[:, 1] == 0) & (data[:, 2] > 0)]

# Create two more axes
axes = [fig.add_subplot(gs[:2, 4:]), fig.add_subplot(gs[2:, 4:])]
# Fit 1/x curve to data
popt, pcov = sciop.curve_fit(func, ratios[:, 0], ratios[:, 3])
xfunc = np.linspace(np.min(ratios[:, 0]), np.max(ratios[:, 0]), 1000)
# Plot all data with curve
axes[0].scatter(
    popt[0] / (1 - popt[1]), 1, color="red", marker="*", s=400, zorder=15, edgecolor="k"
)
axes[0].scatter(ratios[:, 0], ratios[:, 3], marker="o", color="blue", s=2)
axes[0].scatter(infs[:, 0], infs[:, 3], marker="o", color="red", s=2)
axes[0].plot(xfunc, func(xfunc, *popt), color="red", lw=3)
axes[0].hlines(1, np.min(data[:, 0]), np.max(data[:, 0]), color="k", linestyle="dashed")
axes[0].set_xlabel("Empirical Baseline Firing Rate (Hz)")
axes[0].set_ylabel("Inhibited Bin Ratio: ISIF / SDF")
axes[0].set_ylim([0.5, 4])

# Plot histogram of ratios
axes[1].hist(data[:, 3], bins=np.arange(0, 5, 0.25), color="blue", edgecolor="k")
axes[1].set_xlabel("Inhibited Bin Ratio: ISIF / SDF")
axes[1].set_ylabel("Count")
axes[1].set_xlim([0, np.max(data[:, 3])])

# Clean up figure
labels = ["I", "J"]
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


plt.savefig("../figures/figure_5.pdf")
plt.close()
os.system(f"open ../figures/figure_5.pdf")
