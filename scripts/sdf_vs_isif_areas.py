import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys, time, string
import seaborn as sns
from datetime import datetime, timedelta

#sys.path.append('/Users/johnparker/neural_response_classification/python_code')
sys.path.append('/Users/johnparker/streac')

import poisson_spike_train as poisson
#import stimulus_classification as stimclass
import excitation_check as exch
import inhibition_check as inch
from helpers import *

def normalize_fcn(x,t):
    area = np.trapz(x,t)
    return x/area

def find_baseline_areas(spikes,bin_edges,tt,isISIF,shuffles=10,norm=False,avg=250):
    bl_isi = np.diff(spikes) # Find the baseline ISI values
    bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of tts
    bl_spikes = [[spikes[0]+sum(bl_shuff[0:i]) if i > 0 else spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
    bl_isi_fs = [inch.isi_function(bl_spike,tt,avg=avg) for bl_spike in bl_spikes] if isISIF else [exch.kernel(bl_spike,tt,bandwidth=25/1000) for bl_spike in bl_spikes]
    if norm:
        for i in range(len(bl_isi_fs)):
            bl_isi_fs[i] = normalize_fcn(bl_isi_fs[i],tt)
    bl_isif_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
    for i in range(shuffles): # Loop through all the shuffles
        for bi in range(len(bin_edges)-1): # In each shuffle, loop throuhg each bin
            st,ed = np.where(tt <= bin_edges[bi])[0][-1],np.where(tt < bin_edges[bi+1])[0][-1]
            bl_isif_areas.append(np.trapz(bl_isi_fs[i][st:ed+1],x=tt[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
    return bl_isif_areas
            

def find_stim_areas(spikes,bin_edges,tt,isISIF,avg=250,bandwidth=25/1000):
    stim_areas = []
    stim_isif = inch.isi_function(spikes,tt,avg=avg) if isISIF else exch.kernel(spikes,tt,bandwidth=bandwidth)
    for i in range(len(bin_edges)-1): 
        st,ed = np.where(tt <= bin_edges[i])[0][-1],np.where(tt < bin_edges[i+1])[0][-1]
        stim_areas.append(np.trapz(stim_isif[st:ed+1],x=tt[st:ed+1]))
    return np.asarray(stim_areas)

def isif_sdf_ratio(stim_sdf_areas,baseline_sdf_percentile,stim_isif_areas,baseline_isif_percentile):
    num_sdf = np.count_nonzero(stim_sdf_areas < baseline_sdf_percentile)
    num_isif = np.count_nonzero(stim_isif_areas >= baseline_isif_percentile)
    return  num_isif, num_sdf #num_isif / num_sdf if num_sdf > 0 else -1

np.random.seed(24);

high_rate = 30; 
low_rate = 10;
decrease = 50;

t0 = 0; tf = 10; bin_width = 0.5;
sigma = 25/1000;
mu = 250;
tt = np.linspace(0,tf,tf*1000)
bin_edges = np.arange(0,tf+bin_width,bin_width)

#high_spikes = poisson.poisson_spike_train(high_rate,tf)
#low_spikes = poisson.poisson_spike_train(low_rate,tf)
#high_spikes_reduced = poisson.poisson_spike_train(high_rate*(100-decrease)/100,tf)
#low_spikes_reduced = poisson.poisson_spike_train(low_rate*(100-decrease)/100,tf)

high_spikes = poisson.spike_train_generator(high_rate,tf)
low_spikes = poisson.spike_train_generator(low_rate,tf)
high_spikes_reduced = poisson.spike_train_generator(high_rate*(100-decrease)/100,tf)
low_spikes_reduced = poisson.spike_train_generator(low_rate*(100-decrease)/100,tf)

print(high_spikes.shape[0]/tf)
print(high_spikes_reduced.shape[0]/tf)
print(low_spikes.shape[0]/tf)
print(low_spikes_reduced.shape[0]/tf)

sdf_high = exch.kernel(high_spikes,tt,bandwidth=sigma)
isif_high = inch.isi_function(high_spikes,tt,avg=mu)
sdf_low = exch.kernel(low_spikes,tt,bandwidth=sigma)
isif_low = inch.isi_function(low_spikes,tt,avg=mu)

sdf_high_reduced = exch.kernel(high_spikes_reduced,tt,bandwidth=sigma)
isif_high_reduced = inch.isi_function(high_spikes_reduced,tt,avg=mu)
sdf_low_reduced = exch.kernel(low_spikes_reduced,tt,bandwidth=sigma)
isif_low_reduced = inch.isi_function(low_spikes_reduced,tt,avg=mu)


#fig, ax = plt.subplots(5,2,figsize=(16,12),dpi=300)

fig = plt.figure(figsize=(12,8),dpi=300,tight_layout=True)

gs = fig.add_gridspec(4,6)

axes = [fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[2,:2]),fig.add_subplot(gs[3,:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[1,2:4]),fig.add_subplot(gs[2,2:4]),fig.add_subplot(gs[3,2:4])]
#axes = [fig.add_subplot(gs[:2,4:]),fig.add_subplot(gs[2:,4:])]


#axes = [ax[0,0],ax[1,0],ax[2,0],ax[3,0],ax[0,1],ax[1,1],ax[2,1],ax[3,1]]

#fcns_to_plot = [sdf_high,isif_high,1/sdf_high,sdf_high_reduced,isif_high_reduced,1/sdf_high_reduced,sdf_low,isif_low,1/sdf_low,sdf_low_reduced,isif_low_reduced,1/sdf_low_reduced]
spikes_to_plot = [high_spikes,high_spikes_reduced,high_spikes,high_spikes_reduced,low_spikes,low_spikes_reduced,low_spikes,low_spikes_reduced]


params = []

row = 0;
for k in range(len(axes)):
    if (k%4) == 0:
        sdf_areas =  find_baseline_areas(spikes_to_plot[k],bin_edges,tt,False,norm=False)
        sdf_percentile = np.percentile(sdf_areas,99)
        sdf_low_percentile =np.percentile(sdf_areas,1)
        axes[k].hist(sdf_areas,bins=20,alpha=0.5,edgecolor="k",color="blue",label="Shuffled SDF Areas")
        axes[k].vlines(sdf_percentile,0,axes[k].get_ylim()[1],linestyle="dashed",color="k")
        axes[k].vlines(sdf_low_percentile,0,axes[k].get_ylim()[1],linestyle="dotted",color="k")
        params.append(sdf_low_percentile)
        
    elif (k%4) == 1:
        stim_areas = find_stim_areas(spikes_to_plot[k],bin_edges,tt,False,avg=mu,bandwidth=sigma)
        axes[k].bar(list(range(1,len(stim_areas)+1)),stim_areas,color='gray',edgecolor="black",width=1,label="Comparison SDF Areas")
        axes[k].hlines(sdf_percentile,0.5,20.5,color="k",linestyle="dashed")
        axes[k].hlines(sdf_low_percentile,0.5,20.5,color="k",linestyle="dotted")
        axes[k].set_xticks(np.linspace(0,20,5))
        axes[k].set_xticklabels(np.linspace(0,10,5))
        axes[k].set_xlim([0.5,20.5])
        params.append(stim_areas)
    elif (k%4) == 2:
        isif_areas =  find_baseline_areas(spikes_to_plot[k],bin_edges,tt,True,norm=False)
        isif_percentile = np.percentile(isif_areas,99)
        isif_low_percentile =np.percentile(isif_areas,1)
        axes[k].hist(isif_areas,bins=20,alpha=0.5,edgecolor="k",color="blue",label="Shuffled ISIF Areas")
        axes[k].vlines([isif_percentile],0,axes[k].get_ylim()[1],linestyle="dashed",color="k")
        axes[k].vlines([isif_low_percentile],0,axes[k].get_ylim()[1],linestyle="dotted",color="k")
        params.append(isif_percentile)
        
    else:
        stim_areas = find_stim_areas(spikes_to_plot[k],bin_edges,tt,True,avg=mu,bandwidth=sigma)
        axes[k].bar(list(range(1,len(stim_areas)+1)),stim_areas,color='gray',edgecolor="black",width=1,label="Comparison ISIF Areas")
        axes[k].hlines(isif_percentile,0.5,20.5,color="k",linestyle="dashed")
        axes[k].hlines(isif_low_percentile,0.5,20.5,color="k",linestyle="dotted")
        axes[k].set_xticks(np.linspace(0,20,5))
        axes[k].set_xticklabels(np.linspace(0,10,5))
        axes[k].set_xlim([0.5,20.5])
        print(isif_sdf_ratio(params[1],params[0],stim_areas,params[2]))
    #axes[k].legend(edgecolor="gray",fontsize="xx-small",loc="upper right")

axes[0].set_title("High Rate")
axes[4].set_title("Low Rate")

axes[0].set_ylabel("Count")
axes[2].set_ylabel("Count")

axes[1].set_ylabel("SDF Area")
axes[3].set_ylabel("ISIF Area")

makeNice(axes)

'''
sns.despine(fig=fig)
for axe in axes:
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axe.spines[i].set_visible(False)
            axe.tick_params('both', width=0)
        else:
            axe.spines[i].set_linewidth(3)
            axe.tick_params('both', width=0)

labels = string.ascii_uppercase
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")
    if ((i%4) == 0) or ((i%4)==2):
        axes[i].set_xlabel("Area")
    else:
        axes[i].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(f"../figures/sdf_vs_isif_areas.pdf")
plt.close()
#os.system(f"open ../figures/sdf_vs_isif_areas.pdf")
'''

labels = string.ascii_uppercase
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")
    if ((i%4) == 0) or ((i%4)==2):
        axes[i].set_xlabel("Area")
    else:
        axes[i].set_xlabel("Time (s)")



Nrates = 1000; max_rate = 50; min_rate = 5;
random_rates = np.sort(np.random.randint(low=min_rate,high=max_rate,size=(Nrates)))


tf = 100;
t0 = 0; bin_width = 0.5; 
sigma = 25/1000;
mu = 250;
tt = np.linspace(0,tf,tf*1000)
bin_edges = np.arange(0,tf+bin_width,bin_width)
data = np.zeros((Nrates,4))
tot_tt = 0;
tremaining = 0;
'''
sim = 1;
for k in range(Nrates):
    start = time.time()
    rate = random_rates[k]
    #spikes = poisson.poisson_spike_train(rate,tf)
    #spikes_reduced = poisson.poisson_spike_train(rate*(100-decrease)/100,tf)
    spikes = poisson.spike_train_generator(rate,tf)
    spikes_reduced = poisson.spike_train_generator(rate*(100-decrease)/100,tf)
    data[k,0] = spikes.shape[0]/ tf
    baseline_sdf_percentile = np.percentile(find_baseline_areas(spikes,bin_edges,tt,False,norm=False),1)
    stim_sdf_areas = find_stim_areas(spikes_reduced,bin_edges,tt,False,avg=mu,bandwidth=sigma)
    baseline_isif_percentile = np.percentile(find_baseline_areas(spikes,bin_edges,tt,True,norm=False),99)
    stim_isif_areas = find_stim_areas(spikes_reduced,bin_edges,tt,True,avg=mu,bandwidth=sigma)
    num_isif, num_sdf = isif_sdf_ratio(stim_sdf_areas,baseline_sdf_percentile,stim_isif_areas,baseline_isif_percentile)
    data[k,1:3] = [num_isif,num_sdf]
    data[k,3] = num_isif / num_sdf if num_sdf > 0 else num_isif
    end = time.time()
    diff = (end-start);
    tot_tt += diff;
    tremaining = (tot_tt/(sim))*(Nrates - sim);
    est_done = datetime.now()+timedelta(seconds=tremaining)
    trem = time.strftime('%H:%M:%S', time.gmtime(tremaining))
    print(f"Sim: {sim}/{Nrates}, Time: {diff:.02f}, TRemaining: {trem} Est Done: {est_done.ctime()}")
    sim += 1

data = data[data[:, 0].argsort()]
np.savetxt(f"data_tf_{tf}_hz_{min_rate}_{max_rate}_N_{Nrates}.txt",data,newline="\n",delimiter="\t")
'''


data = np.loadtxt(f"data_tf_{tf}_hz_{min_rate}_{max_rate}_N_{Nrates}.txt")
ratios = data[(data[:,2] > 0) & (data[:,1] > 0)]
infs = data[(data[:,2] == 0) & (data[:,1] > 0)]
sdfs = data[(data[:,1] == 0) & (data[:,2] > 0)]


hist_bins = 20;
hist_min = np.floor(np.min(data[:,0]))
hist_max = np.ceil(np.max(data[:,0]))
hist_edges = np.linspace(hist_min,hist_max,hist_bins+1)

hist = np.zeros(hist_edges.shape[0]-1)
for i in range(hist_edges.shape[0]-1):
    hist_data = data[(data[:,0] >= hist_edges[i]) & (data[:,0] < hist_edges[i+1])  & (data[:,2] > 0) & (data[:,1] != data[:,2])]
    hist[i] = (hist_data[(hist_data[:,1] > hist_data[:,2])].shape[0] / hist_data.shape[0])*100 if len(hist_data) > 0 else 0


print(ratios[ratios[:,-1]>1].shape[0],infs.shape[0],(ratios[ratios[:,-1]>1].shape[0]+infs.shape[0])/data[data[:,1] != data[:,2]].shape[0])
print(ratios.shape[0],infs.shape[0],sdfs.shape[0])

axes = [fig.add_subplot(gs[:2,4:]),fig.add_subplot(gs[2:,4:])]
#axes = [ax[4,0],ax[4,1]]

axes[0].scatter(ratios[:,0],ratios[:,3],marker="o",color="blue",s=2)
axes[0].scatter(infs[:,0],infs[:,3],marker="o",color="red",s=2)
axes[0].hlines(1,np.min(data[:,0]),np.max(data[:,0]),color="k",linestyle="dashed")
axes[0].set_xlabel("Empirical Baseline Firing Rate (Hz)")
axes[0].set_ylabel("Inhibited Bin Ratio: ISIF / SDF")
#axes[0].set_xlim([0,30])

#axes[1].bar(hist_edges[:-1],hist,color="gray",edgecolor="k",width=np.mean(np.diff(hist_edges)))
#axes[1].hlines(50,hist_edges[0]-np.mean(np.diff(hist_edges))/2,hist_edges[-1]+np.mean(np.diff(hist_edges))/2,color="k",linestyle="dashed")

axes[1].hist(data[:,3],bins=np.arange(0,5,0.25),color="blue",edgecolor="k")


#axes[1].set_xticks(hist_edges[0::2])
axes[1].set_xlabel("Inhibited Bin Ratio: ISIF / SDF")
axes[1].set_ylabel("Count")
axes[1].set_xlim([0,np.max(data[:,3])])
#axes[1].set_xticklabels([f"{x:.2f}" for x in axes[1].get_xticks()],rotation=45)
#axes[1].set_xlim([0,30])

labels = ["I","J"]
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")

makeNice(axes)
#plt.tight_layout()


plt.savefig("../figures/sdf_vs_isif_areas_combo.pdf")
plt.close()
os.system(f"open ../figures/sdf_vs_isif_areas_combo.pdf")




'''
fig, ax = plt.subplots(1,2,figsize=(8,4),dpi=300)
axes = [ax[0],ax[1]]
axes[0].scatter(ratios[:,0],ratios[:,3],marker="o",color="blue",s=2)
#axes[0].scatter(infs[:,0],infs[:,3],marker="o",color="red",s=2)
axes[0].hlines(1,np.min(data[:,0]),np.max(data[:,0]),color="k",linestyle="dashed")
axes[0].set_xlabel("Empirical Baseline Rate (Hz)")
axes[0].set_ylabel("Inhibited Bin Ratio: ISIF / SDF")
#axes[0].set_xlim([0,30])

axes[1].bar(hist_edges[:-1],hist,color="gray",edgecolor="k",width=np.mean(np.diff(hist_edges)))
axes[1].hlines(50,hist_edges[0],hist_edges[-1],color="k",linestyle="dashed")


axes[1].set_xticks(hist_edges[:-1])
axes[1].set_xlabel("Firing Rate (Hz)")
axes[1].set_ylabel("Percentage ISIF bins > SDF bins")
axes[1].set_xticklabels([f"{x:.2f}" for x in axes[1].get_xticks()],rotation=45)
#axes[1].set_xlim([0,30])

makeNice(axes)
plt.savefig("../figures/multi_runs_sdf_isif_areas.pdf")
plt.close()

os.system(f"open ../figures/multi_runs_sdf_isif_areas.pdf")
'''


