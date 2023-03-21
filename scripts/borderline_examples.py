import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')
#sys.path.append('/Users/johnparker/streac')

def ai_example(csv,cell_num=34,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice"):
    df = pd.read_csv(csv)
    #fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)

    #axes = [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]

    fig = plt.figure(figsize=(10,6),dpi=300)
    gs = fig.add_gridspec(4,2)

    axes = [fig.add_subplot(gs[0:2,0]), fig.add_subplot(gs[0:2,1]), fig.add_subplot(gs[2:,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[3,1])]


    sample = df[(df["delivery"]==delivery) & (df["cell_num"]==cell_num) & (df["neural_response"] == "adapting inhibition") & (df["mouse"] == mouse)]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))

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

    axes[2].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    axes[2].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    axes[2].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    axes[2].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
    axes[2].set_ylabel("Firing Rate (Hz)");
    axes[2].set_xticks(list(range(-10,11,5)))
    axes[2].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[2].set_xlabel("Time (s)")
    sns.despine()
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[2].spines[i].set_visible(False)
            axes[2].tick_params('both', width=0)
        else:
            axes[2].spines[i].set_linewidth(3)
            axes[2].tick_params('both', width=0)


    for trial in range(1,neuron.trials+1):
        bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.txt")
        stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")
        axes[0].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[0].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")

        axes[1].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[1].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")
    axes[0].set_yticks(list(range(1,neuron.trials+1)))
    axes[0].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[0].set_xticks(list(range(-10,11,5)))
    axes[0].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Trial",rotation=90)
    ylims = axes[0].get_ylim()
    axes[0].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[0].set_ylim(ylims)
    axes[0].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[0].spines[i].set_visible(False)
            axes[0].tick_params('both', width=0)
        else:
            axes[0].spines[i].set_linewidth(3)
            axes[0].tick_params('both', width=0)

    axes[1].set_yticks(list(range(1,neuron.trials+1)))
    axes[1].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[1].set_xticks(np.linspace(-1,1,5))
    axes[1].set_xticklabels(np.linspace(-1,1,5),fontsize=8)
    axes[1].set_xlim([-1,1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Trial",rotation=90)
    ylims = axes[1].get_ylim()
    axes[1].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[1].set_ylim(ylims)
    axes[1].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[1].spines[i].set_visible(False)
            axes[1].tick_params('both', width=0)
        else:
            axes[1].spines[i].set_linewidth(3)
            axes[1].tick_params('both', width=0)

    avg_bins = np.loadtxt(f"{neuron.cell_dir}/avg_bin_results.txt")
    bin_edges = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")).bin_edges

    avgsdf = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")))
    for trial in range(1,neuron.trials+1):
        avgsdf += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt") / neuron.trials
    t = np.linspace(0,10,len(avgsdf))
    axes[3].plot(t,avgsdf,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[3].set_ylabel("Average SDF (sps)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_xticks(list(range(0,12,2)))
    axes[3].set_xticklabels(list(range(0,12,2)),fontsize=8)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[3].spines[i].set_visible(False)
            axes[3].tick_params('both', width=0)
        else:
            axes[3].spines[i].set_linewidth(3)
            axes[3].tick_params('both', width=0)

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt") / neuron.trials

    axes[4].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 1:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[4].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[4].set_ylabel("Average ISIF (s)")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_xticks(list(range(0,12,2)))
    axes[4].set_xticklabels(list(range(0,12,2)),fontsize=8)
    sns.despine(fig=fig)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[4].spines[i].set_visible(False)
            axes[4].tick_params('both', width=0)
        else:
            axes[4].spines[i].set_linewidth(3)
            axes[4].tick_params('both', width=0)

    labels = ["A","B","C","D","E"]
    for kk in range(len(axes)):
        axes[kk].text(0.03,0.98,labels[kk],fontsize=16,transform=axes[kk].transAxes,fontweight="bold",color="gray")

    axes[2].text(0.25,0.98,"adapting inhibition",fontsize=16,transform=axes[2].transAxes,fontweight="bold",color="gray")


    plt.tight_layout()
    plt.savefig(f"../figures/borderline_AI.pdf")
    plt.close()

    os.system("open ../figures/borderline_AI.pdf")

def other_example(csv,cell_num=34,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice"):
    types_abbrev = {"complete inhibition":"CI","partial inhibition":"PI","adapting inhibition":"AI","no effect":"NE","excitation":"EX","biphasic IE":"BPIE","biphasic EI":"BPEI"}

    df = pd.read_csv(csv)

    fig = plt.figure(figsize=(10,6),dpi=300)
    gs = fig.add_gridspec(4,2)

    axes = [fig.add_subplot(gs[0:2,0]), fig.add_subplot(gs[0:2,1]), fig.add_subplot(gs[2:,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[3,1])]


    sample = df[(df["delivery"]==delivery) & (df["cell_num"]==cell_num) & (df["mouse"] == mouse)]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
    type =types_abbrev[neuron.neural_response];

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

    axes[2].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    axes[2].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    axes[2].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    axes[2].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
    axes[2].set_ylabel("Firing Rate (Hz)");
    axes[2].set_xticks(list(range(-10,11,5)))
    axes[2].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[2].set_xlabel("Time (s)")
    sns.despine()
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[2].spines[i].set_visible(False)
            axes[2].tick_params('both', width=0)
        else:
            axes[2].spines[i].set_linewidth(3)
            axes[2].tick_params('both', width=0)


    for trial in range(1,neuron.trials+1):
        bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.txt")
        stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")
        axes[0].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[0].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")

        axes[1].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[1].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")
    axes[0].set_yticks(list(range(1,neuron.trials+1)))
    axes[0].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[0].set_xticks(list(range(-10,11,5)))
    axes[0].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Trial",rotation=90)
    ylims = axes[0].get_ylim()
    axes[0].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[0].set_ylim(ylims)
    axes[0].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[0].spines[i].set_visible(False)
            axes[0].tick_params('both', width=0)
        else:
            axes[0].spines[i].set_linewidth(3)
            axes[0].tick_params('both', width=0)

    axes[1].set_yticks(list(range(1,neuron.trials+1)))
    axes[1].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[1].set_xticks(np.linspace(-1,1,5))
    axes[1].set_xticklabels(np.linspace(-1,1,5),fontsize=8)
    axes[1].set_xlim([-1,1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Trial",rotation=90)
    ylims = axes[1].get_ylim()
    axes[1].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[1].set_ylim(ylims)
    axes[1].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[1].spines[i].set_visible(False)
            axes[1].tick_params('both', width=0)
        else:
            axes[1].spines[i].set_linewidth(3)
            axes[1].tick_params('both', width=0)

    avg_bins = np.loadtxt(f"{neuron.cell_dir}/avg_bin_results.txt")
    bin_edges = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")).bin_edges

    avgsdf = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")))
    for trial in range(1,neuron.trials+1):
        avgsdf += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt") / neuron.trials
    t = np.linspace(0,10,len(avgsdf))
    axes[3].plot(t,avgsdf,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[3].set_ylabel("Average SDF (sps)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_xticks(list(range(0,12,2)))
    axes[3].set_xticklabels(list(range(0,12,2)),fontsize=8)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[3].spines[i].set_visible(False)
            axes[3].tick_params('both', width=0)
        else:
            axes[3].spines[i].set_linewidth(3)
            axes[3].tick_params('both', width=0)

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt") / neuron.trials

    axes[4].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 1:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[4].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[4].set_ylabel("Average ISIF (s)")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_xticks(list(range(0,12,2)))
    axes[4].set_xticklabels(list(range(0,12,2)),fontsize=8)
    sns.despine(fig=fig)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[4].spines[i].set_visible(False)
            axes[4].tick_params('both', width=0)
        else:
            axes[4].spines[i].set_linewidth(3)
            axes[4].tick_params('both', width=0)

    labels = ["A","B","C","D","E"]
    for kk in range(len(axes)):
        axes[kk].text(0.03,0.98,labels[kk],fontsize=16,transform=axes[kk].transAxes,fontweight="bold",color="gray")

    axes[2].text(0.25,0.98,f"{neuron.neural_response}",fontsize=16,transform=axes[2].transAxes,fontweight="bold",color="gray")


    plt.tight_layout()
    plt.savefig(f"../figures/borderline_{cell_num}_{type}.pdf")
    plt.close()

    os.system(f"open ../figures/borderline_{cell_num}_{type}.pdf")

def other_example_peri(csv,cell_num=34,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice"):
    types_abbrev = {"complete inhibition":"CI","partial inhibition":"PI","adapting inhibition":"AI","no effect":"NE","excitation":"EX","biphasic IE":"BPIE","biphasic EI":"BPEI"}

    df = pd.read_csv(csv)

    fig = plt.figure(figsize=(10,6),dpi=300)
    gs = fig.add_gridspec(4,2)

    axes = [fig.add_subplot(gs[0:2,0]), fig.add_subplot(gs[0:2,1]), fig.add_subplot(gs[2:,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[3,1])]


    sample = df[(df["delivery"]==delivery) & (df["cell_num"]==cell_num) & (df["mouse"] == mouse)]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
    type =types_abbrev[neuron.neural_response];

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

    axes[1].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    axes[1].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    axes[1].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    axes[1].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
    axes[1].set_ylabel("Firing Rate (Hz)");
    axes[1].set_xticks(list(range(-10,11,5)))
    axes[1].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[1].set_xlabel("Time (s)")
    sns.despine()
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[1].spines[i].set_visible(False)
            axes[1].tick_params('both', width=0)
        else:
            axes[1].spines[i].set_linewidth(3)
            axes[1].tick_params('both', width=0)

    bw = 0.5
    psth = np.zeros(int(20/bw))
    psth_edges = np.linspace(-10,10,int(20/bw)+1)
    
    for trial in range(1,neuron.trials+1):
        bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.txt")
        stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")
        axes[0].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[0].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")

        combo = np.sort(np.append(stim,bl-10))
        for k in range(len(psth)):
            psth[k] += len(combo[(combo >= psth_edges[k]) & (combo < psth_edges[k+1])])
            
    
    axes[0].set_yticks(list(range(1,neuron.trials+1)))
    axes[0].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[0].set_xticks(list(range(-10,11,5)))
    axes[0].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Trial",rotation=90)
    ylims = axes[0].get_ylim()
    axes[0].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[0].set_ylim(ylims)
    axes[0].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[0].spines[i].set_visible(False)
            axes[0].tick_params('both', width=0)
        else:
            axes[0].spines[i].set_linewidth(3)
            axes[0].tick_params('both', width=0)

    avg_bins = np.loadtxt(f"{neuron.cell_dir}/avg_bin_results.txt")
    bin_edges = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")).bin_edges

    avgsdf = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")))
    for trial in range(1,neuron.trials+1):
        avgsdf += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt") / neuron.trials
    t = np.linspace(0,10,len(avgsdf))
    axes[3].plot(t,avgsdf,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[3].set_ylabel("Average SDF (sps)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_xticks(list(range(0,12,2)))
    axes[3].set_xticklabels(list(range(0,12,2)),fontsize=8)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[3].spines[i].set_visible(False)
            axes[3].tick_params('both', width=0)
        else:
            axes[3].spines[i].set_linewidth(3)
            axes[3].tick_params('both', width=0)

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt") / neuron.trials

    axes[4].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 1:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[4].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[4].set_ylabel("Average ISIF (s)")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_xticks(list(range(0,12,2)))
    axes[4].set_xticklabels(list(range(0,12,2)),fontsize=8)

    axes[2].bar(np.linspace(-10,10,int(20/bw)),height=psth,width=bw,edgecolor="k",color="blue",alpha=0.75)
    ylims = axes[2].get_ylim()
    axes[2].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Spike Count")
    axes[2].set_xticks(list(range(-10,11,5)))
    axes[2].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    sns.despine(fig=fig)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[4].spines[i].set_visible(False)
            axes[4].tick_params('both', width=0)
            axes[2].spines[i].set_visible(False)
            axes[2].tick_params('both', width=0)
        else:
            axes[4].spines[i].set_linewidth(3)
            axes[4].tick_params('both', width=0)
            axes[2].spines[i].set_linewidth(3)
            axes[2].tick_params('both', width=0)

    labels = ["A","B","C","D","E"]
    for kk in range(len(axes)):
        axes[kk].text(0.03,0.98,labels[kk],fontsize=16,transform=axes[kk].transAxes,fontweight="bold",color="gray")

    axes[1].text(0.25,0.98,f"{neuron.neural_response}",fontsize=16,transform=axes[1].transAxes,fontweight="bold",color="gray")
    
    
    

    plt.tight_layout()
    plt.savefig(f"../figures/borderline_{cell_num}_{type}_peri.pdf")
    plt.close()

    


    os.system(f"open ../figures/borderline_{cell_num}_{type}_peri.pdf")

    plt.figure()





save_direc= "/Users/johnparker/neural_response_classification/Data/PV_Hsyn_DD_Naive/Results_fixed_isif"
delivery = "PV-DIO-ChR2 in GPe"
csv = f"{save_direc}/comparisons/all_data.csv"
types = ["complete inhibition","partial inhibition","adapting inhibition","no effect","excitation","biphasic IE","biphasic EI"]
types_abbrev = ["CI","PI","AI","NE","EX","BPIE","BPEI"]

cell_nums = [39,99,34,22,78,89,73]
delivery = ["hsyn-ChR2 in GPe","PV-DIO-ChR2 in GPe","PV-DIO-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe"]
mouse = ["Naive mice","Naive mice","6-OHDA mice","Naive mice","6-OHDA mice","Naive mice","Naive mice"]

ai_example(csv)
other_example(csv,cell_num=34,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=81,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=72,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice")

other_example_peri(csv,cell_num=34,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=81,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=72,delivery="PV-DIO-ChR2 in GPe",mouse="6-OHDA mice")


other_example(csv,cell_num=91,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=98,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=85,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=93,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=57,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=50,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=100,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=75,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=11,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example(csv,cell_num=14,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")

other_example_peri(csv,cell_num=91,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=98,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=85,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=93,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=57,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=50,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=100,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=75,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=11,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")
other_example_peri(csv,cell_num=14,delivery="hsyn-ChR2 in GPe",mouse="6-OHDA mice")

sys.exit()

# to send: bpie 14, ex 11, 75 ne, 93 ne, 98 ne

# AI Naive Candidates: 114 hsyn, 100 hsyn, 11 hsyn, 38 pv, 48 pv
# AI 6-OHDA Candidates: 31 pv
# AI 34 PV DD should be the match


# NE Naive candidates:
# NE 6-OHDA Candidates: 91 hysn, 98 hsyn, 85 hsyn, 93 hsyn, 57 hsyn, 50 hsyn, 100 hsyn, 75 hsyn

# EX Naive candidates:
# EX DD Candidates: 11 hsyn, 72 pv

# BPIE 6-OHDA candidates: 14 hysn, 81 hsyn
df = pd.read_csv(csv)

for ii in range(1,len(delivery)):
    print(cell_nums[ii],delivery[ii],types[ii],mouse[ii])

    fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)

    axes = [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]

    sample = df[(df["delivery"]==delivery[ii]) & (df["cell_num"]==cell_nums[ii]) & (df["neural_response"] == types[ii]) & (df["mouse"] == mouse[ii])]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))

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

    axes[1].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    axes[1].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    axes[1].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    axes[1].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
    axes[1].set_ylabel("Firing Rate (Hz)");
    axes[1].set_xticks(list(range(-10,11,5)))
    axes[1].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[1].set_xlabel("Time (s)")
    sns.despine()
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[1].spines[i].set_visible(False)
            axes[1].tick_params('both', width=0)
        else:
            axes[1].spines[i].set_linewidth(3)
            axes[1].tick_params('both', width=0)


    for trial in range(1,neuron.trials+1):
        bl = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.txt")
        stim = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")
        axes[0].scatter(bl-10,np.ones(len(bl))*trial,marker="|",s=50,color="k")
        axes[0].scatter(stim,np.ones(len(stim))*trial,marker="|",s=50,color="k")
    axes[0].set_yticks(list(range(1,neuron.trials+1)))
    axes[0].set_yticklabels(list(range(1,neuron.trials+1)))
    axes[0].set_xticks(list(range(-10,11,5)))
    axes[0].set_xticklabels(list(range(-10,11,5)),fontsize=8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Trial",rotation=90)
    ylims = axes[0].get_ylim()
    axes[0].vlines(0,ylims[0],ylims[1],color="gray",linestyle="dashed")
    axes[0].set_ylim(ylims)
    axes[0].add_patch(patches.Rectangle((0,neuron.trials+0.2),10,0.2,color="r",facecolor="r"))
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[0].spines[i].set_visible(False)
            axes[0].tick_params('both', width=0)
        else:
            axes[0].spines[i].set_linewidth(3)
            axes[0].tick_params('both', width=0)

    avg_bins = np.loadtxt(f"{neuron.cell_dir}/avg_bin_results.txt")
    bin_edges = pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")).bin_edges


    avgsdf = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt")))
    for trial in range(1,neuron.trials+1):
        avgsdf += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/sdf.txt") / neuron.trials
    t = np.linspace(0,10,len(avgsdf))
    avgsdf = avgsdf / np.max(avgsdf)
    axes[2].plot(t,avgsdf,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[2].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[2].set_ylabel("Average SDF")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0,0.5,1])
    axes[2].set_yticklabels([0,0.5,1],fontsize=8)
    axes[2].set_xticks(list(range(0,12,2)))
    axes[2].set_xticklabels(list(range(0,12,2)),fontsize=8)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[2].spines[i].set_visible(False)
            axes[2].tick_params('both', width=0)
        else:
            axes[2].spines[i].set_linewidth(3)
            axes[2].tick_params('both', width=0)

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/isif.txt") / neuron.trials

    avgisif = avgisif / np.max(avgisif)
    axes[3].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        if avg_bins[bi] == 1:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[3].set_ylabel("Average Trial ISIF")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yticks([0,0.5,1])
    axes[3].set_yticklabels([0,0.5,1],fontsize=8)
    axes[3].set_xticks(list(range(0,12,2)))
    axes[3].set_xticklabels(list(range(0,12,2)),fontsize=8)
    sns.despine(fig=fig)
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axes[3].spines[i].set_visible(False)
            axes[3].tick_params('both', width=0)
        else:
            axes[3].spines[i].set_linewidth(3)
            axes[3].tick_params('both', width=0)

    labels = ["A","B","C","D"]
    for kk in range(len(axes)):
        axes[kk].text(0.03,0.98,labels[kk],fontsize=16,transform=axes[kk].transAxes,fontweight="bold",color="gray")

    axes[1].text(0.25,0.98,types[ii],fontsize=16,transform=axes[1].transAxes,fontweight="bold",color="gray")


    plt.tight_layout()
    plt.savefig(f"../figures/borderline_{types_abbrev[ii]}.pdf")
    plt.close()
