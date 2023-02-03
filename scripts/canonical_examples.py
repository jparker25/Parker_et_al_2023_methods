import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')

save_direc= "/Users/johnparker/neural_response_classification/Data/PV_Hsyn_DD_Naive/Results"
delivery = "PV-DIO-ChR2 in GPe"
csv = f"{save_direc}/comparisons/all_data.csv"
types = ["complete inhibition", "partial inhibition","adapting inhibition", "excitation","biphasic IE", "biphasic EI","no effect"]

cell_nums = [106,17,110,78,92,66,49]
delivery = ["hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","PV-DIO-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","PV-DIO-ChR2 in GPe","PV-DIO-ChR2 in GPe"]

cell_nums = [39,105,90,49,78,89,73]
delivery = ["hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","PV-DIO-ChR2 in GPe","PV-DIO-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe"]
mouse = ["Naive mice","Naive mice","6-OHDA mice","6-OHDA mice","6-OHDA mice","Naive mice","Naive mice"]
types = ["complete inhibition","partial inhibition","adapting inhibition","no effect","excitation","biphasic IE","biphasic EI"]

df = pd.read_csv(csv)

#fig, axes = plt.subplots(7,1,figsize=(8,6),dpi=300)

fig = plt.figure(figsize=(10,8),dpi=300)
gs = fig.add_gridspec(8,4)

axes = [fig.add_subplot(gs[0:2, :2]),fig.add_subplot(gs[2:4, :2]),fig.add_subplot(gs[4:6, :2]),fig.add_subplot(gs[0:2, 2:]),fig.add_subplot(gs[2:4, 2:]),fig.add_subplot(gs[4:6, 2:]),fig.add_subplot(gs[6:, 1:3])]

for i in range(7):
    ax = axes[i]

    #sample = df[(df["neural_response"] == types[i])].sample()

    sample = df[(df["delivery"]==delivery[i]) & (df["cell_num"]==cell_nums[i]) & (df["neural_response"] == types[i]) & (df["mouse"]==mouse[i])]
    print(sample)
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
    #print(neuron.cell_num,neuron.delivery,neuron.mouse,neuron.neural_response)

    baselines = []; stimuli = [];
    for trial in range(1,neuron.trials+1):
        baselineFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
        baselines.append(pickle.load(baselineFile))
        baselineFile.close()
        stimuliFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")
        stimuli.append(pickle.load(stimuliFile))
        stimuliFile.close()
    avg_bl = np.zeros((len(baselines[0].bins),2))
    avg_stim = np.zeros((len(stimuli[0].bins),2))

    for baseline in baselines:
        freq = np.zeros(()); cv = []
        for bi in range(len(baseline.bins)):
            avg_bl[bi,0] += baseline.bins[bi].freq/len(baselines)
            avg_bl[bi,1] += baseline.bins[bi].cv/len(baselines)
    for stimulus in stimuli:
        for bi in range(len(stimulus.bins)):
            avg_stim[bi,0] += stimulus.bins[bi].freq/len(stimuli)
            avg_stim[bi,1] += stimulus.bins[bi].cv/len(stimuli)
    yerr_bl = np.zeros((len(baselines[0].bins),len(baselines),2))
    for bi in range(len(baselines[0].bins)):
        for bl in range(len(baselines)):
            yerr_bl[bi,bl,0] = baselines[bl].bins[bi].freq
    yerr_stim = np.zeros((len(stimuli[0].bins),len(stimuli),2))
    for bi in range(len(stimuli[0].bins)):
        for bl in range(len(stimuli)):
            yerr_stim[bi,bl,0] = stimuli[bl].bins[bi].freq

    ax.plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    ax.plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    ax.fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    ax.fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
    ax.set_ylabel("Firing Rate",fontsize=14);
    ax.set_yticklabels([int(x) for x in ax.get_yticks()],fontsize=14)
    ax.set_xticks(list(range(-10,11,5)))
    ax.set_xticklabels(list(range(-10,11,5)),fontsize=14)
    ax.set_xlabel("Time (s)",fontsize=14)
    sns.despine()
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            ax.spines[i].set_visible(False)
            ax.tick_params('both', width=0)
        else:
            ax.spines[i].set_linewidth(3)
            ax.tick_params('both', width=0)

labels = ["A","B","C","D","E","F","G"]
for i in range(len(axes)):
    #axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")
    axes[i].text(0.25,0.98,types[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")

plt.tight_layout()
plt.savefig(f"../figures/canonical_examples.pdf")
plt.close()

os.system(f"open ../figures/canonical_examples.pdf")