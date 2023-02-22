import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')

import poisson_spike_train as poisson
import stimulus_classification as stimclass

def normalize_fcn(x,t):
    area = np.trapz(x,t)
    return x/area

save_direc= "/Users/johnparker/neural_response_classification/Data/PV_Hsyn_DD_Naive/Results_fixed"
delivery = "PV-DIO-ChR2 in GPe"
csv = f"{save_direc}/comparisons/all_data.csv"
types = ["complete inhibition","partial inhibition","adapting inhibition","no effect","excitation","biphasic IE","biphasic EI"]
types_abbrev = ["CI","PI","AI","NE","EX","BPIE","BPEI"]

cell_nums = [39,99,114,22,78,89,73]
delivery = ["hsyn-ChR2 in GPe","PV-DIO-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe","hsyn-ChR2 in GPe"]
mouse = ["Naive mice","Naive mice","Naive mice","Naive mice","6-OHDA mice","Naive mice","Naive mice"]

np.random.seed(24);

rate = 25; t0 = 0; tf = 10;
sigma = 25;
mu = 25;

spikes = poisson.poisson_spike_train(rate,tf,t_start=0)
#spikes = np.append(spikes,[1.5,2,2.01,2.5,4.5])
#spikes = np.sort(spikes)
#spikes = np.asarray([0.5,3, 7, 9])

time = np.linspace(0,tf,tf*1000)
sdf = stimclass.kernel(spikes,time,bandwidth=sigma/1000)
sdf75 = stimclass.kernel(spikes,time,bandwidth=sigma*3/1000)
sdf250 = stimclass.kernel(spikes,time,bandwidth=sigma*10/1000)
isif = stimclass.isi_function(spikes,time,avg=mu)
isif250 = stimclass.isi_function(spikes,time,avg=mu*10)

fig, ax = plt.subplots(1,3,figsize=(8,3),dpi=300)
axes = [ax[0],ax[1],ax[2]]
#axes[0].scatter(spikes,np.ones(spikes.shape[0]),marker="|",s=200,color="k")

axes[0].plot(time,sdf,color="blue",label="$\sigma=25$ms")
axes[0].plot(time,sdf250,color="red",label="$\sigma = 250$ms")
axes[0].scatter(spikes,np.ones(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[0].legend(edgecolor="gray",fontsize="xx-small")
axes[0].set_ylabel("SDF (sps)")

axes[1].plot(time,isif,color="red",label="$\mu=25$ms")
axes[1].plot(time,isif250,color="blue",label="$\mu=250$ms")
axes[1].scatter(spikes,np.zeros(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[1].legend(edgecolor="gray",fontsize="xx-small")
axes[1].set_ylabel("ISIF")

axes[2].plot(time,1/sdf75,color="green",linestyle="dashed",label="1/SDF, $\sigma=75$ms")
axes[2].plot(time,1/sdf250,color="red",label="1/SDF, $\sigma=250$ms")
axes[2].plot(time,isif250,color="blue",label="ISIF, $\mu=250$ms",alpha=0.75)

axes[2].scatter(spikes,np.zeros(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[2].legend(edgecolor="gray",fontsize="xx-small")

print(np.trapz(isif250,time)/np.trapz(isif,time))
print(np.trapz(sdf250,time)/np.trapz(sdf,time))
print(np.trapz(isif250,time)/np.trapz(1/sdf250,time))
print(np.trapz(isif,time)/np.trapz(1/sdf,time))
print(np.mean(sdf),np.mean(sdf250))
print(np.mean(isif),np.mean(isif250),np.mean(np.diff(spikes)))

sns.despine(fig=fig)
for axe in axes:
    for i in ['left','right','top','bottom']:
        if i != 'left' and i != 'bottom':
            axe.spines[i].set_visible(False)
            axe.tick_params('both', width=0)
        else:
            axe.spines[i].set_linewidth(3)
            axe.tick_params('both', width=0)

'''
for i in ['left','right','top','bottom']:
    axes[0].spines[i].set_visible(False)
    axes[0].tick_params('both', width=0)
    axes[0].set_yticks([])
'''

labels = ["A","B","C","D","E","F","G"]
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")
    axes[i].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("../figures/sdf_isif_parameter_comparison.pdf")
plt.close()

os.system("open ../figures/sdf_isif_parameter_comparison.pdf")