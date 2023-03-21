import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import string
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')
#sys.path.append('/Users/johnparker/streac')

import poisson_spike_train as poisson
import stimulus_classification as stimclass

def normalize_fcn(x,t):
    area = np.trapz(x,t)
    return x/area

def find_areas(spikes,bin_edges,time,isISIF,shuffles=10,norm=True):
    bl_isi = np.diff(spikes) # Find the baseline ISI values
    bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of times
    bl_spikes = [[spikes[0]+sum(bl_shuff[0:i]) if i > 0 else spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
    bl_isi_fs = [stimclass.isi_function(bl_spike,time,avg=250) for bl_spike in bl_spikes] if isISIF else [stimclass.kernel(bl_spike,time,bandwidth=25/1000) for bl_spike in bl_spikes]
    if norm:
        for i in range(len(bl_isi_fs)):
            bl_isi_fs[i] = normalize_fcn(bl_isi_fs[i],time)
    bl_isif_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
    for i in range(shuffles): # Loop through all the shuffles
        for bi in range(len(bin_edges)-1): # In each shuffle, loop throuhg each bin
            st,ed = np.where(time <= bin_edges[bi])[0][-1],np.where(time < bin_edges[bi+1])[0][-1]
            bl_isif_areas.append(np.trapz(bl_isi_fs[i][st:ed+1],x=time[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
    return bl_isif_areas
            

np.random.seed(24);

high_rate = 30; 
low_rate = 10;
decrease = 10;

t0 = 0; tf = 5; bin_width = 0.5;
sigma = 25;
mu = 250;
time = np.linspace(0,tf,tf*1000)
bin_edges = np.arange(0,tf+bin_width,bin_width)


high_spikes = poisson.poisson_spike_train(high_rate,tf,t_start=0)
low_spikes = poisson.poisson_spike_train(low_rate,tf)
high_spikes_reduced = poisson.poisson_spike_train(high_rate*(100-decrease)/100,tf)
low_spikes_reduced = poisson.poisson_spike_train(low_rate*(100-decrease)/100,tf)

sdf_high = stimclass.kernel(high_spikes,time,bandwidth=sigma/1000)
isif_high = stimclass.isi_function(high_spikes,time,avg=mu)
sdf_low = stimclass.kernel(low_spikes,time,bandwidth=sigma/1000)
isif_low = stimclass.isi_function(low_spikes,time,avg=mu)

sdf_high_reduced = stimclass.kernel(high_spikes_reduced,time,bandwidth=sigma/1000)
isif_high_reduced = stimclass.isi_function(high_spikes_reduced,time,avg=mu)
sdf_low_reduced = stimclass.kernel(low_spikes_reduced,time,bandwidth=sigma/1000)
isif_low_reduced = stimclass.isi_function(low_spikes_reduced,time,avg=mu)



fig, ax = plt.subplots(3,4,figsize=(12,8),dpi=300)

axes = [ax[0,0],ax[1,0],ax[2,0],ax[0,1],ax[1,1],ax[2,1],ax[0,2],ax[1,2],ax[2,2],ax[0,3],ax[1,3],ax[2,3]]

fcns_to_plot = [sdf_high,isif_high,1/sdf_high,sdf_high_reduced,isif_high_reduced,1/sdf_high_reduced,sdf_low,isif_low,1/sdf_low,sdf_low_reduced,isif_low_reduced,1/sdf_low_reduced]
spikes_to_plot = [high_spikes,high_spikes_reduced,low_spikes,low_spikes_reduced]

title=["SDF","ISIF","1/SDF"]

row = 0;
for k in range(len(axes)):
    if k%3 == 0 and k!= 0:
        row += 1
    if row == 0:
        axes[k].set_title(title[k])
    axes[k].plot(time,fcns_to_plot[k],color="blue",label=f"Rate ${spikes_to_plot[row].shape[0]/tf}$Hz")
    axes[k].scatter(spikes_to_plot[row],np.zeros(spikes_to_plot[row].shape[0]),s=50,marker="|",color="k")
    axes[k].legend(edgecolor="gray",fontsize="xx-small",loc="upper right")
    

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
    axes[i].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(f"../figures/sdf_vs_isif_{mu}ms_mix.pdf")
plt.close()



fig, ax = plt.subplots(3,4,figsize=(12,8),dpi=300)

axes = [ax[0,0],ax[1,0],ax[2,0],ax[0,1],ax[1,1],ax[2,1],ax[0,2],ax[1,2],ax[2,2],ax[0,3],ax[1,3],ax[2,3]]

fcns_to_plot = [sdf_high,isif_high,1/sdf_high,sdf_high_reduced,isif_high_reduced,1/sdf_high_reduced,sdf_low,isif_low,1/sdf_low,sdf_low_reduced,isif_low_reduced,1/sdf_low_reduced]
spikes_to_plot = [high_spikes,high_spikes_reduced,low_spikes,low_spikes_reduced]

title=["SDF","ISIF","Areas"]

row = 0;
for k in range(len(axes)):
    if k%3 == 0 and k!= 0:
        row += 1
    if row == 0:
        axes[k].set_title(title[k])
    if (k % 3) == 2:
        sdf_areas =  find_areas(spikes_to_plot[row],bin_edges,time,False,norm=True)
        isif_areas =  find_areas(spikes_to_plot[row],bin_edges,time,True,norm=True)
        hist_edges = np.linspace(0,np.max([np.max(sdf_areas),np.max(isif_areas)]),21)
        sdf_percentile = np.percentile(sdf_areas,99)
        sdf_low_percentile =np.percentile(sdf_areas,1)
        isif_percentile = np.percnetile(isif_areas,99)
        axes[k].hist(sdf_areas,bins=hist_edges,alpha=0.5,label="SDF",edgecolor="k",color="blue")
        axes[k].hist(isif_areas,bins=hist_edges,alpha=0.5,color="red",label="ISIF",edgecolor="k")
        axes[k].vlines
        axes[k].legend(edgecolor="gray",fontsize="xx-small",loc="upper right")
    else:
        axes[k].plot(time,fcns_to_plot[k],color="blue",label=f"Rate ${spikes_to_plot[row].shape[0]/tf}$Hz")
        axes[k].scatter(spikes_to_plot[row],np.zeros(spikes_to_plot[row].shape[0]),s=50,marker="|",color="k")
        axes[k].legend(edgecolor="gray",fontsize="xx-small",loc="upper right")
    

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
    if i % 3 != 2:
        axes[i].set_xlabel("Time (s)")
    else:
        axes[i].set_xlabel("Normalized Area")

plt.tight_layout()
plt.savefig(f"../figures/sdf_vs_isif_{mu}ms_mix_areas.pdf")
plt.close()




os.system(f"open ../figures/sdf_vs_isif_{mu}ms_mix_areas.pdf")
os.system(f"open ../figures/sdf_vs_isif_{mu}ms.pdf")


