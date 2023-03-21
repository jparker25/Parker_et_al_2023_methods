import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')
#sys.path.append('/Users/johnparker/streac')

def norm_fcn(x,t):
    return x / np.trapz(x,t)

save_direc= "/Users/johnparker/neural_response_classification/Data/PV_Hsyn_DD_Naive/Results_fixed_isif"
delivery = "PV-DIO-ChR2 in GPe"
csv = f"{save_direc}/comparisons/all_data.csv"

df = pd.read_csv(csv)

#fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)
fig = plt.figure(figsize=(8,6),dpi=300)
#gs = fig.add_gridspec(4,4)
gs = fig.add_gridspec(3,4)
sample = df[(df["neural_response"] != "no effect") & (df["delivery"]==delivery)].sample()
print(sample["cell_num"])

#axes = [fig.add_subplot(gs[:2, :2]),fig.add_subplot(gs[2:3,0:2]),fig.add_subplot(gs[2:3,2:]),fig.add_subplot(gs[3:4,0:2]),fig.add_subplot(gs[3:,2:]),fig.add_subplot(gs[0:2,2:])]
#axes = [fig.add_subplot(gs[:2, :2]),fig.add_subplot(gs[0,2:]),fig.add_subplot(gs[1,2:]),fig.add_subplot(gs[2:,0:2]),fig.add_subplot(gs[2:,2:])]
axes = [fig.add_subplot(gs[:2, :2]),fig.add_subplot(gs[0,2:]),fig.add_subplot(gs[1,2:]),fig.add_subplot(gs[2,0:2]),fig.add_subplot(gs[2,2:])]

#cell_num = 17
#cell_num = 58
cell_num = 114
sample = df[(df["cell_num"]==cell_num) & (df["neural_response"] != "no effect") & (df["delivery"]==delivery)]
cell_dir = sample["cell_dir"].values[0]
neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))

ax = axes[0]
for trial in range(1,neuron.trials+1):
    bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.txt")
    stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")
    ax.scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
    ax.scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")

ax.set_yticks(list(range(1,neuron.trials+1)))
ax.set_yticklabels(list(range(1,neuron.trials+1)))
ax.set_xticks(list(range(-10,11,5)))
ax.set_xticklabels(list(range(-10,11,5)),fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Trial Number",rotation=90)
ylims = ax.get_ylim()
ax.vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
ax.set_ylim(ylims)
ax.add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))

sns.despine(fig=fig)
for i in ['left','right','top','bottom']:
    if i != 'left' and i != 'bottom':
        ax.spines[i].set_visible(False)
        ax.tick_params('both', width=0)
    else:
        ax.spines[i].set_linewidth(3)
        ax.tick_params('both', width=0)

ax = axes[1]
for trial in range(1,neuron.trials+1):
    stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")
    stim = stim/np.max(stim)
    ax.plot(np.linspace(0,10,len(stim)),stim+trial)
ax.set_yticks(list(range(1,neuron.trials+1)))
ax.set_yticklabels(list(range(1,neuron.trials+1)),fontsize=8)
ax.set_ylabel("Trial # SDF (sps)",rotation=90)
ax.set_xlabel("Time (s)")
ax.set_xticks(list(range(0,12,2)))
ax.set_xticklabels(list(range(0,12,2)),fontsize=8)
sns.despine(fig=fig)
for i in ['left','right','top','bottom']:
    if i != 'left' and i != 'bottom':
        ax.spines[i].set_visible(False)
        ax.tick_params('both', width=0)
    else:
        ax.spines[i].set_linewidth(3)
        ax.tick_params('both', width=0)

ax = axes[2]
for trial in range(1,neuron.trials+1):
    #bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/sdf.txt")
    isif = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")
    isif = isif/np.max(isif)
    #ax.plot(np.linspace(-10,0,len(bl)),bl)
    ax.plot(np.linspace(0,10,len(isif)),isif+trial)
ax.set_yticks(list(range(1,neuron.trials+1)))
ax.set_yticklabels(list(range(1,neuron.trials+1)),fontsize=8)
ax.set_ylabel("Trial # ISIF (s)",rotation=90)
ax.set_xlabel("Time (s)")
ax.set_xticks(list(range(0,12,2)))
ax.set_xticklabels(list(range(0,12,2)),fontsize=8)
sns.despine(fig=fig)
for i in ['left','right','top','bottom']:
    if i != 'left' and i != 'bottom':
        ax.spines[i].set_visible(False)
        ax.tick_params('both', width=0)
    else:
        ax.spines[i].set_linewidth(3)
        ax.tick_params('both', width=0)

avg_bins = np.loadtxt(f"{neuron.cell_dir}/avg_bin_results.txt")
bin_edges = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")).bin_edges

ax = axes[3]
avgsdf = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")))
for trial in range(1,neuron.trials+1):
    avgsdf += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt") / neuron.trials
t = np.linspace(0,10,len(avgsdf))
ax.plot(t,avgsdf,color="blue")
for bi in range(len(bin_edges)-1):
    if avg_bins[bi] == 2:
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        ax.fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
ax.set_ylabel("Average SDF (sps)")
ax.set_xlabel("Time (s)")
ax.set_xticks(list(range(0,12,2)))
ax.set_xticklabels(list(range(0,12,2)),fontsize=8)

sns.despine(fig=fig)
for i in ['left','right','top','bottom']:
    if i != 'left' and i != 'bottom':
        ax.spines[i].set_visible(False)
        ax.tick_params('both', width=0)
    else:
        ax.spines[i].set_linewidth(3)
        ax.tick_params('both', width=0)

ax = axes[4]
avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")))
for trial in range(1,neuron.trials+1):
    avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt") / neuron.trials

ax.plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
for bi in range(len(bin_edges)-1):
    if avg_bins[bi] == 1:
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        ax.fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
ax.set_ylabel("Average ISIF")
ax.set_xlabel("Time (s)")
ax.set_xticks(list(range(0,12,2)))
ax.set_xticklabels(list(range(0,12,2)),fontsize=8)
sns.despine(fig=fig)
for i in ['left','right','top','bottom']:
    if i != 'left' and i != 'bottom':
        ax.spines[i].set_visible(False)
        ax.tick_params('both', width=0)
    else:
        ax.spines[i].set_linewidth(3)
        ax.tick_params('both', width=0)

labels = ["A","B","C","D","E","F"]
axlabel = [axes[0],axes[1],axes[2],axes[3],axes[4]]
for i in range(len(axlabel)):
    axlabel[i].text(0.03,0.98,labels[i],fontsize=16,transform=axlabel[i].transAxes,fontweight="bold",color="gray")

plt.tight_layout()
plt.savefig(f"../figures/average_example.pdf")
plt.close()

os.system(f"open ../figures/average_example.pdf")
