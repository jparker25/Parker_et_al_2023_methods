"""
Generate Figure 4.

Usage: python figure_4.py
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle, os, sys
import seaborn as sns

# Change to the absolute path to the STReaC toolbox
sys.path.append("/path/to/streac")

import excitation_check as exch
import inhibition_check as inch

# Read in example data
save_direc = "../methods_results"
delivery = "PV-DIO-ChR2 in GPe"
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

# Find example and read in example trial
sample = df[
    (df["group"] == "Naive_mice_hsyn-ChR2_in_GPe")
    & (df["cell_num"] == 76)
    & (df["mouse"] == "Naive mice")
]
cell_dir = sample["cell_dir"].values[0]
neuron = pickle.load(open(f"{cell_dir}/neuron.obj", "rb"))
trial = 9
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

# Set randomness
shuffles = 10
percentile = 99
np.random.seed(24)

# Generate figure
fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300)
axes = [ax[0, 0], ax[1, 0], ax[2, 0], ax[0, 1], ax[1, 1], ax[2, 1]]

# Look at baseline data and find excitation in stimulus
### EXCITATION ###
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
bl_sdf_areas = (
    []
)  # Empty array that will store the areas of each bin of the shuffled baseline
bl_sdf_fs = [exch.kernel(bl_spike, baseline_train.t) for bl_spike in bl_spikes]
stim_areas = []
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
for i in range(len(stim_data_train.bin_edges) - 1):  # Iterate through each bin
    st, ed = (
        np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],
        np.where(stim_data_train.t < stim_data_train.bin_edges[i + 1])[0][-1],
    )
    stim_areas.append(
        np.trapz(stim_data_train.sdf[st : ed + 1], x=stim_data_train.t[st : ed + 1])
    )
    if np.percentile(bl_sdf_areas, percentile) < np.trapz(
        stim_data_train.sdf[st : ed + 1], x=stim_data_train.t[st : ed + 1]
    ):  # See if area of bin is beyond the percentile of the baseline shuffled areas
        axes[2].fill_between(
            stim_data_train.t[st : ed + 1],
            stim_data_train.sdf[st : ed + 1],
            color="blue",
            alpha=0.5,
        )  # Fill the area of the SDF that is beyond the percentile of the baseline areas
        results[
            i
        ] += 1  # Make the classification of the bin a 1 indicating it is excited

# Create a histogram of all the baseline SDF areas
axes[0].hist(bl_sdf_areas, bins=20, color="gray", edgecolor="black")
axes[0].vlines(
    np.percentile(bl_sdf_areas, percentile),
    axes[0].get_ylim()[0],
    axes[0].get_ylim()[1],
    color="k",
    linestyle="dashed",
)  # Draw a verticle line where the percentile threshold is on the histogram
axes[0].set_xlabel("Baseline SDF Area Bins")
axes[0].set_ylabel("Count")  # Label the histogram axes
axes[0].text(40, 31, "99th Percentile", fontsize=8, fontweight="bold", color="gray")
axes[0].annotate(
    "",
    xy=(np.percentile(bl_sdf_areas, percentile), 28),
    xytext=(40, 32),
    arrowprops=dict(arrowstyle="-|>", color="k"),
)

# Show bar heights for stimulus areas
axes[1].bar(
    list(range(1, len(stim_areas) + 1)),
    stim_areas,
    color="gray",
    edgecolor="black",
    width=1,
)
axes[1].hlines(
    np.percentile(bl_sdf_areas, percentile),
    1,
    len(stim_areas),
    color="k",
    linestyle="dashed",
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Stimulus SDF Area")
axes[1].set_xlim([0.5, 20.5])
axes[1].set_xticks(np.arange(0.5, 24.5, 4))
axes[1].set_xticklabels(np.arange(0, 12, 2))

# Plot SDF as function of t and shade excited areas
axes[2].plot(
    stim_data_train.t, stim_data_train.sdf, color="blue", label="Stim ISI Interpolation"
)
axes[2].scatter(
    stim_data_train.spikes, np.zeros(len(stim_data_train.spikes)), marker="|", color="k"
)

# Label
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("SDF (sps)")  # Label the SDF axes
axes[2].set_xlim([0, 10])


# Look at baseline data and find inhibition in stimulus
### INHIBITION ###
results = np.zeros(
    len(stim_data_train.bin_edges) - 1
)  # Create empty array that will store the classifications of each bin
isi_stim = stim_data_train.isif  # Create SDF for baseline and stimulus spikes
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
stim_areas = []
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
first_bin_edge = np.where(stim_data_train.bin_edges <= stim_data_train.spikes[0])[0][
    -1
]  # Find the bin edge that is before the first stimulus spike
last_bin_edge = np.where(stim_data_train.bin_edges < stim_data_train.spikes[-1])[0][
    -1
]  # Find the bin edge that is before the last stimulus spike

for i in range(len(stim_data_train.bin_edges) - 1):  # Iterate through each bin
    if i == first_bin_edge:
        st, ed = (
            np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],
            np.where(stim_data_train.t < stim_data_train.bin_edges[i + 1])[0][-1],
        )
        isi_stim_bin = stim_data_train.isif[st : ed + 1]  # Find the SDF of the bin
        if stim_data_train.spikes[0] >= np.percentile(bl_isif_areas, percentile):
            axes[5].fill_between(
                stim_data_train.t[0:st],
                stim_data_train.isif[0:st],
                color="blue",
                alpha=0.5,
            )
            results[0:i] += 1
        if np.percentile(bl_isif_areas, percentile) < np.trapz(
            isi_stim_bin, x=stim_data_train.t[st : ed + 1]
        ):  # See if area of bin is beyond the percentile of the baseline shuffled areas
            axes[5].fill_between(
                stim_data_train.t[st : ed + 1], isi_stim_bin, color="blue", alpha=0.5
            )  # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[
                i
            ] += 1  # Make the classification of the bin a 1 indicating it is excited
        stim_areas.append(np.trapz(isi_stim_bin, x=stim_data_train.t[st : ed + 1]))
    elif i == last_bin_edge:
        st, ed = (
            np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],
            np.where(stim_data_train.t < stim_data_train.bin_edges[i + 1])[0][-1],
        )
        isi_stim_bin = stim_data_train.isif[st : ed + 1]  # Find the SDF of the bin
        stim_areas.append(np.trapz(isi_stim_bin, x=stim_data_train.t[st : ed + 1]))
        if (
            stim_data_train.time - stim_data_train.spikes[-1]
            >= np.percentile(bl_isif_areas, percentile)
            and i != len(stim_data_train.bin_edges) - 2
        ):
            results[i + 1 :] += 1
            axes[5].fill_between(
                stim_data_train.t[ed + 1 :],
                stim_data_train.isif[ed + 1 :],
                color="blue",
                alpha=0.5,
            )
        if np.percentile(bl_isif_areas, percentile) < np.trapz(
            isi_stim_bin, x=stim_data_train.t[st : ed + 1]
        ):  # See if area of bin is beyond the percentile of the baseline shuffled areas
            axes[5].fill_between(
                stim_data_train.t[st : ed + 1], isi_stim_bin, color="blue", alpha=0.5
            )  # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[
                i
            ] += 1  # Make the classification of the bin a 1 indicating it is excited
    elif i > first_bin_edge and i < last_bin_edge:
        st, ed = (
            np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],
            np.where(stim_data_train.t < stim_data_train.bin_edges[i + 1])[0][-1],
        )
        isi_stim_bin = stim_data_train.isif[st : ed + 1]  # Find the SDF of the bin
        if np.percentile(bl_isif_areas, percentile) < np.trapz(
            isi_stim_bin, x=stim_data_train.t[st : ed + 1]
        ):  # See if area of bin is beyond the percentile of the baseline shuffled areas
            axes[5].fill_between(
                stim_data_train.t[st : ed + 1], isi_stim_bin, color="blue", alpha=0.5
            )  # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[
                i
            ] += 1  # Make the classification of the bin a 1 indicating it is excited
        stim_areas.append(np.trapz(isi_stim_bin, x=stim_data_train.t[st : ed + 1]))

# Create a histogram of all the baseline ISIF areas
axes[3].hist(bl_isif_areas, bins=20, color="gray", edgecolor="black")
axes[3].vlines(
    np.percentile(bl_isif_areas, percentile),
    axes[3].get_ylim()[0],
    axes[3].get_ylim()[1],
    color="k",
    linestyle="dashed",
)  # Draw a verticle line where the percentile threshold is on the histogram
axes[3].set_xlabel("Baseline ISI Area Bins")
axes[3].set_ylabel("Count")  # Label the histogram axes
axes[3].text(0.0115, 38, "99th Percentile", fontsize=8, fontweight="bold", color="gray")
axes[3].annotate(
    "",
    xy=(np.percentile(bl_isif_areas, percentile), 34),
    xytext=(0.014, 37),
    arrowprops=dict(arrowstyle="-|>", color="k"),
)

# Plot ISIF stimulus areas and compare with baseline
axes[4].bar(
    list(range(1, len(stim_areas) + 1)),
    stim_areas,
    color="gray",
    edgecolor="black",
    width=1,
)
axes[4].hlines(
    np.percentile(bl_isif_areas, percentile),
    1,
    len(stim_areas),
    color="k",
    linestyle="dashed",
)
axes[4].set_xlabel("Time (s)")
axes[4].set_ylabel("Stimulus ISIF Area")
axes[4].set_xlim([0.5, 20.5])
axes[4].set_xticks(np.arange(0.5, 24.5, 4))
axes[4].set_xticklabels(np.arange(0, 12, 2))

# Plot the ISIF as a function of t and shade inhibited areas
axes[5].plot(
    stim_data_train.t,
    stim_data_train.isif,
    color="blue",
    label="Stim ISI Interpolation",
)
axes[5].scatter(
    stim_data_train.spikes, np.zeros(len(stim_data_train.spikes)), marker="|", color="k"
)
axes[5].set_xlabel("Time(s)")
axes[5].set_ylabel("ISIF (s)")  # Label the SDF axes
axes[5].set_xlim([0, 10])


# Figure clean up
sns.despine(fig=fig)
for axe in axes:
    for i in ["left", "right", "top", "bottom"]:
        if i != "left" and i != "bottom":
            axe.spines[i].set_visible(False)
            axe.tick_params("both", width=0)
        else:
            axe.spines[i].set_linewidth(3)
            axe.tick_params("both", width=0)


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
plt.savefig("../figures/figure_4.pdf")
plt.close()

os.system("open ../figures/figure_4.pdf")
