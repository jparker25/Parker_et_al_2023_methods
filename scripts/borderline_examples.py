import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/streac')

def other_example(csv,cell_num=34,group="6-OHDA_mice_hsyn_ChR2_in_GPe"):
    types_abbrev = {"complete inhibition":"CI","partial inhibition":"PI","adapting inhibition":"AI","no effect":"NE","excitation":"EX","biphasic IE":"BPIE","biphasic EI":"BPEI"}

    df = pd.read_csv(csv)

    fig = plt.figure(figsize=(10,6),dpi=300)
    gs = fig.add_gridspec(4,2)

    axes = [fig.add_subplot(gs[0:2,0]), fig.add_subplot(gs[0:2,1]), fig.add_subplot(gs[2:,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[3,1])]
    sample = df[(df["group"]==group) & (df["cell_num"]==cell_num)]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
    light_on = np.loadtxt(f"{cell_dir}/light_on.txt")
    type =types_abbrev[neuron.neural_response];

    baselines = []; stimuli = [];
    binsize = 0.25;
    all_fr = []; bins = np.arange(-10,10+binsize,binsize)
    for trial in range(1,neuron.trials+1):
        baselineFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
        baselines.append(pickle.load(baselineFile))
        baselineFile.close()
        stimuliFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")
        stimuli.append(pickle.load(stimuliFile))
        stimuliFile.close()
        trial_spikes = neuron.spikes[(neuron.spikes >= light_on[trial-1]-10) & (neuron.spikes < light_on[trial-1]+10)] - light_on[trial-1]
        fr,_= np.histogram(trial_spikes,bins=bins)
        all_fr.append(fr / binsize)
    
    avg_bl = np.zeros((len(baselines[0].bins),2))
    avg_stim = np.zeros((len(stimuli[0].bins),2))

    all_average = np.asarray(all_fr)
    if all_average.shape[0] > 0:
        mean_all_average = np.mean(all_average,axis=0)
        sem = np.std(all_average,axis=0)
        plot_bins = bins[:-1]

        axes[2].plot(plot_bins[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0],color="black",linewidth=2)
        axes[2].plot(plot_bins[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0],color="blue",linewidth=2)
        axes[2].fill_between(plot_bins[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0]+sem[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0]-sem[bins[:-1] <= 0],color="black",edgecolor="black",alpha=0.5)
        axes[2].fill_between(plot_bins[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0]+sem[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0]-sem[bins[:-1] >= 0],color="blue",edgecolor="blue",alpha=0.5)


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

    #axes[2].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    #axes[2].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    #axes[2].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    #axes[2].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
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
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        axes[3].vlines(t[st],0,avgsdf[st],linestyle="dotted",color="gray",alpha=0.5)
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[3].vlines(t[ed],0,avgsdf[ed],linestyle="dotted",color="gray",alpha=0.5)
    axes[3].hlines(2*neuron.excitation_threshold,0,10,linestyle="dashed",color="gray")
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

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/{'isif' if neuron.isif_inhibition else 'sdf'}.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/{'isif' if neuron.isif_inhibition else 'sdf'}.txt") / neuron.trials

    axes[4].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        axes[4].vlines(t[st],0,avgisif[st],linestyle="dotted",color="gray",alpha=0.5)
        if avg_bins[bi] == 1:
            axes[4].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[4].vlines(t[ed],0,avgisif[ed],linestyle="dotted",color="gray",alpha=0.5)
    axes[4].hlines(2*neuron.inhibition_threshold,0,10,linestyle="dashed",color="gray")
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

def other_example_peri(csv,cell_num=34,group="6-OHDA_mice_hsyn_ChR2_in_GPe"):
    types_abbrev = {"complete inhibition":"CI","partial inhibition":"PI","adapting inhibition":"AI","no effect":"NE","excitation":"EX","biphasic IE":"BPIE","biphasic EI":"BPEI"}

    df = pd.read_csv(csv)

    fig = plt.figure(figsize=(10,6),dpi=300)
    gs = fig.add_gridspec(4,2)

    axes = [fig.add_subplot(gs[0:2,0]), fig.add_subplot(gs[0:2,1]), fig.add_subplot(gs[2:,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[3,1])]


    sample = df[(df["group"]==group) & (df["cell_num"]==cell_num)]
    cell_dir = sample["cell_dir"].values[0]
    neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
    light_on = np.loadtxt(f"{cell_dir}/light_on.txt")
    type =types_abbrev[neuron.neural_response];

    baselines = []; stimuli = [];
    binsize = 0.25;
    all_fr = []; bins = np.arange(-10,10+binsize,binsize)
    for trial in range(1,neuron.trials+1):
        baselineFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
        baselines.append(pickle.load(baselineFile))
        baselineFile.close()
        stimuliFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")
        stimuli.append(pickle.load(stimuliFile))
        stimuliFile.close()
        trial_spikes = neuron.spikes[(neuron.spikes >= light_on[trial-1]-10) & (neuron.spikes < light_on[trial-1]+10)] - light_on[trial-1]
        fr,_= np.histogram(trial_spikes,bins=bins)
        all_fr.append(fr / binsize)
    
    avg_bl = np.zeros((len(baselines[0].bins),2))
    avg_stim = np.zeros((len(stimuli[0].bins),2))

    all_average = np.asarray(all_fr)
    if all_average.shape[0] > 0:
        mean_all_average = np.mean(all_average,axis=0)
        sem = np.std(all_average,axis=0)
        plot_bins = bins[:-1]

        axes[1].plot(plot_bins[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0],color="black",linewidth=2)
        axes[1].plot(plot_bins[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0],color="blue",linewidth=2)
        axes[1].fill_between(plot_bins[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0]+sem[bins[:-1] <= 0],mean_all_average[bins[:-1] <= 0]-sem[bins[:-1] <= 0],color="black",edgecolor="black",alpha=0.5)
        axes[1].fill_between(plot_bins[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0]+sem[bins[:-1] >= 0],mean_all_average[bins[:-1] >= 0]-sem[bins[:-1] >= 0],color="blue",edgecolor="blue",alpha=0.5)

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

    #axes[1].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
    #axes[1].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
    #axes[1].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
    #axes[1].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")
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
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        axes[3].vlines(t[st],0,avgsdf[st],linestyle="dotted",color="gray",alpha=0.5)
        if avg_bins[bi] == 2:
            st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
            axes[3].fill_between(t[st:ed+1],avgsdf[st:ed+1],color="blue",alpha=0.5)
    axes[3].vlines(t[ed],0,avgsdf[ed],linestyle="dotted",color="gray",alpha=0.5)
    axes[3].hlines(2*neuron.excitation_threshold,0,10,linestyle="dashed",color="gray")
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

    avgisif = np.zeros(len(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/{'isif' if neuron.isif_inhibition else 'sdf'}.txt")))
    for trial in range(1,neuron.trials+1):
        avgisif += np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/{'isif' if neuron.isif_inhibition else 'sdf'}.txt") / neuron.trials

    axes[4].plot(np.linspace(0,10,len(avgisif)),avgisif,color="blue")
    for bi in range(len(bin_edges)-1):
        st,ed = np.where(t <= bin_edges[bi])[0][-1],np.where(t < bin_edges[bi+1])[0][-1]
        axes[4].vlines(t[st],0,avgisif[st],linestyle="dotted",color="gray",alpha=0.5)
        if avg_bins[bi] == 1:
            axes[4].fill_between(t[st:ed+1],avgisif[st:ed+1],color="blue",alpha=0.5)
    axes[4].vlines(t[ed],0,avgisif[ed],linestyle="dotted",color="gray",alpha=0.5)
    axes[4].hlines(2*neuron.inhibition_threshold,0,10,linestyle="dashed",color="gray")
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



save_direc = "/Users/johnparker/Parker_et_al_2023_methods/methods_results"
csv = f"{save_direc}/all_data.csv"


other_example_peri(csv,cell_num=85,group="6-OHDA_mice_hsyn-ChR2_in_GPe") # NE
other_example_peri(csv,cell_num=16,group="6-OHDA_mice_hsyn-ChR2_in_GPe") # BPIE
other_example_peri(csv,cell_num=82,group="6-OHDA_mice_hsyn-ChR2_in_GPe") # AI
other_example_peri(csv,cell_num=79,group="6-OHDA_mice_hsyn-ChR2_in_GPe") # PI
