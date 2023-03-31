import numpy as np
from matplotlib import lines,pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

#sys.path.append('/Users/johnparker/neural_response_classification/python_code')
sys.path.append('/Users/johnparker/streac')

#import stimulus_classification as stimclass
import excitation_check as exch
import inhibition_check as inch


#save_direc= "/Users/johnparker/neural_response_classification/Data/PV_Hsyn_DD_Naive/Results_fixed_isif"
save_direc = "/Users/johnparker/streac/results/gpe_pv_baseline_stimulus"
delivery = "PV-DIO-ChR2 in GPe"
csv = f"{save_direc}/all_data.csv"
types = ["complete inhibition", "partial inhibition","adapting inhibition", "excitation","biphasic IE", "biphasic EI","no effect"]

df = pd.read_csv(csv)

sample = df[(df["delivery"]=="hsyn-ChR2 in GPe") & (df["cell_num"]==62) & (df["mouse"]=="Naive mice")]

cell_dir = sample["cell_dir"].values[0]
neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
trial = 9;
stim_data_train = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb"))
baseline_train = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb"))
shuffles = 10; percentile= 99;

np.random.seed(24);

### INHIBITION ###
results = np.zeros(len(stim_data_train.bin_edges)-1) # Create empty array that will store the classifications of each bin
isi_stim = stim_data_train.isif# Create SDF for baseline and stimulus spikes
bl_isi = np.diff(baseline_train.spikes) # Find the baseline ISI values
bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of times
bl_spikes = [[baseline_train.spikes[0]+sum(bl_shuff[0:i]) if i > 0 else baseline_train.spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
bl_isi_fs = [inch.isi_function(bl_spike,baseline_train.t) for bl_spike in bl_spikes]
stim_areas = [];
bl_isif_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
for i in range(shuffles): # Loop through all the shuffles
    for bi in range(len(baseline_train.bin_edges)-1): # In each shuffle, loop throuhg each bin
        st,ed = np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],np.where(baseline_train.t < baseline_train.bin_edges[bi+1])[0][-1]
        bl_isif_areas.append(np.trapz(bl_isi_fs[i][st:ed+1],x=baseline_train.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
first_bin_edge = np.where(stim_data_train.bin_edges <= stim_data_train.spikes[0])[0][-1] # Find the bin edge that is before the first stimulus spike
last_bin_edge = np.where(stim_data_train.bin_edges < stim_data_train.spikes[-1])[0][-1] # Find the bin edge that is before the last stimulus spike

for i in range(len(stim_data_train.bin_edges)-1): # Iterate through each bin
    if i == first_bin_edge:
        st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
        isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
        if stim_data_train.spikes[0] >= np.percentile(bl_isif_areas,percentile):
            results[0:i] += 1
        if np.percentile(bl_isif_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
        stim_areas.append(np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]))
    elif i == last_bin_edge:
        st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
        isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
        stim_areas.append(np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]))
        if stim_data_train.time-stim_data_train.spikes[-1] >= np.percentile(bl_isif_areas,percentile) and i != len(stim_data_train.bin_edges)-2:
            results[i+1:] += 1
        if np.percentile(bl_isif_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
    elif i > first_bin_edge and i < last_bin_edge:
        st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
        isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
        if np.percentile(bl_isif_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
        stim_areas.append(np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]))



### EXCITATION ###
results = np.zeros(len(stim_data_train.bin_edges)-1) # Create empty array that will store the classifications of each bin
bl_isi = np.diff(baseline_train.spikes) # Find the baseline ISI values
bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of times
bl_spikes = [[baseline_train.spikes[0]+sum(bl_shuff[0:i]) if i > 0 else baseline_train.spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
bl_sdf_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
bl_sdf_fs = [exch.kernel(bl_spike,baseline_train.t) for bl_spike in bl_spikes]
stim_areas = [];
for i in range(shuffles): # Loop through all the shuffles
    for bi in range(len(baseline_train.bin_edges)-1): # In each shuffle, loop throuhg each bin
        st,ed = np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],np.where(baseline_train.t < baseline_train.bin_edges[bi+1])[0][-1]
        #bl_sdf_areas.append(np.trapz(baseline_train.sdf[st:ed+1],x=baseline_train.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
        bl_sdf_areas.append(np.trapz(bl_sdf_fs[i][st:ed+1],x=baseline_train.t[st:ed+1]))
for i in range(len(stim_data_train.bin_edges)-1): # Iterate through each bin
    st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
    stim_areas.append(np.trapz(stim_data_train.sdf[st:ed+1],x=stim_data_train.t[st:ed+1]))
    if np.percentile(bl_sdf_areas,percentile) < np.trapz(stim_data_train.sdf[st:ed+1],x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
        results[i] += 1 # Make the classification of the bin a 1 indicating it is excited

fig = plt.figure(figsize=(8,4),dpi=300)
gs = fig.add_gridspec(2,3)

axes = [fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]

count = 1;
for shuff in bl_spikes:
    axes[0].scatter(shuff,np.ones(len(shuff))*count,marker="|",s=50,color="k")
    axes[1].plot(baseline_train.t,(bl_sdf_fs[count-1]/np.max(bl_sdf_fs[count-1]))+count,color="blue")
    axes[2].plot(baseline_train.t,(bl_isi_fs[count-1]/np.max(bl_isi_fs[count-1]))+count,color="blue")
    count += 1

for i in range(3):
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Permutation Number")
    axes[i].set_yticks(np.arange(1,11))
    axes[i].set_yticklabels(np.arange(0,10),fontsize=8)


axes[3].hist(bl_sdf_areas,bins=20,color="gray",edgecolor="black") # Create a histogram of all the baseline SDF areas
axes[3].vlines(np.percentile(bl_sdf_areas,percentile),axes[3].get_ylim()[0],axes[3].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
axes[3].set_xlabel("Baseline SDF Area Bins"); axes[3].set_ylabel("Count") # Label the histogram axes


axes[4].hist(bl_isif_areas,bins=20,color="gray",edgecolor="black") # Create a histogram of all the baseline SDF areas
axes[4].vlines(np.percentile(bl_isif_areas,percentile),axes[4].get_ylim()[0],axes[4].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
axes[4].set_xlabel("Baseline ISIF Area Bins"); axes[4].set_ylabel("Count") # Label the histogram axes

sns.despine(fig=fig)

for axe in axes:
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axe.spines[i].set_visible(False)
            axe.tick_params('both', width=0)
        else:
            axe.spines[i].set_linewidth(3)
            axe.tick_params('both', width=0)
    
labels = ["A","B","C","D","E","F","G"]
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")

plt.tight_layout()
plt.savefig("../figures/baseline_shuffling.pdf")
plt.close()

os.system("open ../figures/baseline_shuffling.pdf")
