import numpy as np
from matplotlib import pyplot as plt
import os,sys
from scipy import signal
from scipy.fft import fft, fftfreq

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

np.random.seed(24);

rate = 25; t0 = 0; tf = 10;
sigma = 25;
mu = 25;

#spikes = poisson.poisson_spike_train(rate,tf,t_start=0)
spikes = poisson.spike_train_generator(rate,tf,t_start=0)

time = np.linspace(0,tf,tf*1000)
sdf = exch.kernel(spikes,time,bandwidth=sigma/1000)
sdf75 = exch.kernel(spikes,time,bandwidth=sigma*3/1000)
sdf250 = exch.kernel(spikes,time,bandwidth=sigma*10/1000)
isif = inch.isi_function(spikes,time,avg=mu)
isif250 = inch.isi_function(spikes,time,avg=mu*10)

'''
tspikes = poisson.spike_train_generator(rate,tf,t_start=0)
mmean = inch.isi_function(tspikes,time,avg=mu)
plt.figure()
plt.plot(time,mmean)
plt.scatter(tspikes,np.zeros(tspikes.shape[0]),marker="|",s=50,color="k")
plt.show()
sys.exit()
'''

fig = plt.figure(dpi=300,figsize=(12,4),tight_layout=True)
gs = fig.add_gridspec(2,4)

axes = [fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[:,2]),fig.add_subplot(gs[:,3])]

axes[0].plot(time,sdf,color="blue",label="$\sigma=25$ms")
axes[0].plot(time,sdf250,color="red",label="$\sigma = 250$ms")
axes[0].scatter(spikes,np.ones(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[0].legend(edgecolor="gray",fontsize="xx-small",loc="upper right")
axes[0].set_ylabel("SDF (sps)")
axes[0].set_xlabel("Time (s)")

#f, pxx = signal.periodogram(sdf,1e3)
#f250, pxx250 = signal.periodogram(sdf250,1e3)

#freqs = np.fft.fftfreq(tf*1000,1/1e3)
#idx = np.argsort(freqs)
#pxx = np.abs(np.fft.fft(sdf-np.mean(sdf)))**2
#pxx250 = np.abs(np.fft.fft(sdf250-np.mean(sdf250)))**2


pxx = (2/sdf.shape[0])*np.abs(fft(sdf-np.mean(sdf))[1:sdf.shape[0]//2])**2
pxx250 = (2/sdf250.shape[0])*np.abs(fft(sdf250-np.mean(sdf250))[1:sdf250.shape[0]//2])**2
freqs = fftfreq(sdf.shape[0],1e-3)[1:sdf.shape[0]//2]
#print(max(freqs))


axes[1].semilogy(freqs,pxx,color="blue",label="$\sigma=25$ms")
axes[1].semilogy(freqs,pxx250,color="red",label="$\sigma=250$ms")
axes[1].semilogy(freqs,pxx/pxx250,color="gray",label="Ratio")

'''
#sdf = np.sin(2*np.pi*25*time)+np.random.rand(time.shape[0])
f, pxx = signal.welch(sdf,1e3,nperseg=sdf.shape[0],scaling='spectrum')
f, pxx250 = signal.welch(sdf250,1e3,nperseg=sdf250.shape[0],scaling='spectrum')

axes[1].semilogy(f,pxx,color="blue",label="$\sigma=25$ms")
axes[1].semilogy(f,pxx250,color="red",label="$\sigma=250$ms")
axes[1].semilogy(f,pxx/pxx250,color="gray",label="Ratio")
'''


axes[1].legend(edgecolor="gray",fontsize="xx-small")


axes[1].set_xlim([freqs[1],50])
axes[1].set_ylim([10e-3,10e6])
axes[1].set_ylabel("Power")




#f, pxx = signal.periodogram(isif,1e3)
#f250, pxx250 = signal.periodogram(isif250,1e3)

#pxx = np.abs(np.fft.fft(isif-np.mean(isif)))**2
#pxx250 = np.abs(np.fft.fft(isif250-np.mean(isif250)))**2

pxx = (2/isif.shape[0])*np.abs(fft(isif-np.mean(isif))[1:isif.shape[0]//2])**2
pxx250 = (2/isif250.shape[0])*np.abs(fft(isif250-np.mean(isif250))[1:isif250.shape[0]//2])**2
freqs = fftfreq(isif.shape[0],1e-3)[1:isif.shape[0]//2]

axes[2].semilogy(freqs,pxx,color="red",label="$\mu=25$ms")
axes[2].semilogy(freqs,pxx250,color="blue",label="$\mu=250$ms")
axes[2].semilogy(freqs,pxx250/pxx,color="gray",label="Ratio")
axes[2].set_xlim([freqs[1],50])
#axes[2].set_ylim([10e-9,10e3])
axes[2].set_ylabel("Power")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].legend(edgecolor="gray",fontsize="xx-small")
'''


#sdf = np.sin(2*np.pi*25*time)+np.random.rand(time.shape[0])
f, pxx = signal.welch(isif,1e3,nperseg=isif.shape[0],scaling='spectrum')
f, pxx250 = signal.welch(isif250,1e3,nperseg=isif250.shape[0],scaling='spectrum')

axes[2].semilogy(f,pxx,color="red",label="$\mu=25$ms")
axes[2].semilogy(f,pxx250,color="blue",label="$\mu=250$ms")
axes[2].semilogy(f,pxx250/pxx,color="gray",label="Ratio")
axes[2].legend(edgecolor="gray",fontsize="xx-small")

axes[2].set_xlim([0,50])
#axes[1].set_ylim([10e-9,10e3])
axes[2].set_ylabel("Power")
axes[2].set_xlabel("Frequency (Hz)")
'''

axes[3].plot(time,isif,color="red",label="$\mu=25$ms")
axes[3].plot(time,isif250,color="blue",label="$\mu=250$ms")
axes[3].scatter(spikes,np.zeros(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[3].legend(edgecolor="gray",fontsize="xx-small")
axes[3].set_ylabel("ISIF (s)")
axes[3].set_xlabel("Time (s)")

#axes[4].plot(time,1/sdf75,color="green",linestyle="dashed",label="1/SDF, $\sigma=75$ms")
axes[4].plot(time,1/sdf250,color="red",label="1/SDF, $\sigma=250$ms")
axes[4].plot(time,isif250,color="blue",label="ISIF, $\mu=250$ms",alpha=0.75)

axes[4].scatter(spikes,np.zeros(spikes.shape[0])*-1,marker="|",s=50,color="k")
axes[4].legend(edgecolor="gray",fontsize="xx-small")
axes[4].set_xlabel("Time (s)")
axes[4].set_ylim([axes[3].get_ylim()[0],axes[3].get_ylim()[1]])


labels = ["A","B","C","D","E"]
for i in range(len(axes)):
    axes[i].text(0.03,0.98,labels[i],fontsize=16,transform=axes[i].transAxes,fontweight="bold",color="gray")

makeNice(axes)
plt.savefig("../figures/sdf_isif_parameter_comparison.pdf")
plt.close()

os.system("open ../figures/sdf_isif_parameter_comparison.pdf")