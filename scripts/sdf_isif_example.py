import numpy as np
from matplotlib import lines,pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle,os,sys
import seaborn as sns

sys.path.append('/Users/johnparker/neural_response_classification/python_code')

import stimulus_classification as stimclass

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

sample = df[(df["delivery"]==delivery[3]) & (df["cell_num"]==cell_nums[3]) & (df["neural_response"] == types[3]) & (df["mouse"]==mouse[3])]
print(sample)
cell_dir = sample["cell_dir"].values[0]
neuron = pickle.load(open(f"{cell_dir}/neuron.obj","rb"))
trial = 5;
train = np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.txt")

#fig, ax = plt.subplots(3,2,figsize=(8,6),dpi=300)
#axes = [ax[0,0],ax[1,0],ax[2,0],ax[0,1],ax[1,1],ax[2,1]]

fig = plt.figure(figsize=(8,6),dpi=300)
gs = fig.add_gridspec(4,4)
axes = [fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[2,0:2]),fig.add_subplot(gs[3,0:2]),fig.add_subplot(gs[1,2:]),fig.add_subplot(gs[2,2:]),fig.add_subplot(gs[3,2:])]

axes[0].scatter(train,np.ones(len(train)),marker="|",s=400,color="k")
for i in ['left','right','top','bottom']:
    axes[0].spines[i].set_visible(False)
    axes[0].tick_params('both', width=0)
axes[0].set_yticks([]); #axes[0].set_yticks([]);
rect_sdf = patches.Rectangle((2.25,0.9),0.5,0.2,fill=False,color="gray",linewidth=2)
axes[0].add_patch(rect_sdf)
rect_isi = patches.Rectangle((7.5,0.9),0.5,0.2,fill=False,color="gray",linewidth=2)
axes[0].add_patch(rect_isi)
axes[0].set_ylim([0.8,1.2])
axes[0].set_xlabel("Time (s)")

sdfA = [2.375,0.9]; isifA = [7.625,0.9]
sdfB = [2.5,15]; isifB = [7.75,0.2]
arrow_sdf = patches.ConnectionPatch(sdfA,sdfB,coordsA=axes[0].transData,coordsB=axes[1].transData,color="gray", arrowstyle="-|>", mutation_scale=20,  linewidth=2)
arrow_isif = patches.ConnectionPatch(isifA,isifB,coordsA=axes[0].transData,coordsB=axes[4].transData,color="gray", arrowstyle="-|>", mutation_scale=20,  linewidth=2)
fig.patches.append(arrow_sdf)
fig.patches.append(arrow_isif)




time = np.linspace(0,10,10*1000)
bandwidth = 25/1000;
for spike in train:
    gaus = stimclass.gaussian(time-spike,bandwidth)
    axes[1].plot(time,gaus,color="blue")
    axes[2].plot(time,gaus,color="gray")
sdf = stimclass.kernel(train,time)
axes[2].plot(time,sdf,color="blue")
axes[1].set_xlim([2.25,2.75])
axes[2].set_xlim([2.25,2.75])
axes[1].scatter(train,np.ones(len(train))*-2,marker="|",s=50,color="k")
axes[2].scatter(train,np.ones(len(train))*-5,marker="|",s=50,color="k")

axes[3].scatter(train,np.ones(len(train))*-5,marker="|",s=50,color="k")
axes[3].plot(time,sdf,color="blue")



axes[4].scatter(train,np.zeros(len(train))*-0.2,marker="|",s=50,color="k")
axes[4].plot(train[:-1],np.diff(train),color="blue")
axes[4].set_xlim([7.5,8])

axes[5].scatter(train,np.zeros(len(train))*-2,marker="|",s=50,color="k")
isif = stimclass.isi_function(train,time,avg=250)
axes[5].plot(train[:-1],np.diff(train),color="gray")
axes[5].plot(time,isif,color="blue")
axes[5].set_xlim([7.5,8])

axes[6].scatter(train,np.zeros(len(train))*-2,marker="|",s=50,color="k")
axes[6].plot(time,isif,color="blue")

rect_sdf = patches.Rectangle((2.25,0),0.5,40,fill=False,color="gray",linewidth=2)
axes[3].add_patch(rect_sdf)
rect_isi = patches.Rectangle((7.5,0.015),0.5,0.1,fill=False,color="gray",linewidth=2)
axes[6].add_patch(rect_isi)

for axe in axes[1:]:
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
    axes[i].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("../figures/sdf_isif_example.pdf")
plt.close()

os.system(f"open ../figures/sdf_isif_example.pdf")

