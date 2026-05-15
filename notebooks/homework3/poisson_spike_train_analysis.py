import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

r = 94 #spikes per second
Fs = 1e3 #Sampling frequency
totalTime = 30 #seconds
dt = 0.001 #seconds
binSize = 0.01 #seconds


def readPoiSpikes(fileName, Fs):
    
    data = loadmat(fileName)
    rawspike = np.array(data['spikes'])
    rawspike = rawspike.flatten()
    
    totalTime = int(np.max(rawspike))
    
    if rawspike.size == 0:
        raise ValueError("Empty file")
    
    spikeBins = np.arange(0,totalTime+1,1) #bins in ms
    spikeTrain,_= np.histogram(rawspike, bins=spikeBins)
    spikeTrain = (spikeTrain>0).astype(int)
        
    return spikeTrain


def generatePoiSpikes(r, dt, totalSize):
    p = r*dt
    rand = np.random.rand(totalSize)
    spikeTrain = np.zeros(totalSize)
    spikeTrain = (rand<p).astype(int)

    return spikeTrain


def calcCV(spikeTrain):
    
    spikeIndices = np.where(spikeTrain == 1)[0]
    isi = np.diff(spikeIndices)
    
    if len(isi) < 2:
        return np.nan  # Return NaN if there aren't enough spikes to calculate CV
    
    CV = np.std(isi) / np.mean(isi)
    
    return CV


def calcFF(spikeTrain):
    
    FF = np.var(spikeTrain) / np.mean(spikeTrain) if np.mean(spikeTrain) > 0 else np.nan
    
    return FF


def calcRate(spikeTrain, window, dt):
    
    binWidth = int(window / dt) if window > 0 else len(spikeTrain)
    rateOfFire = [
        np.sum(spikeTrain[i:i+binWidth]) / (binWidth * dt)
        for i in range(0, len(spikeTrain), binWidth)
    ]
    
    if window > 0:
        timeBins = np.arange(0, len(rateOfFire)) * window
        plt.figure(figsize=(12, 3))
        plt.plot(timeBins, rateOfFire)
        plt.xlabel('Time (s)')
        plt.ylabel('Rate of Fire (Hz)')
        plt.title(f'Firing Rate over Time, Window: {window} s')
        plt.grid()
        plt.show()
    
    return np.array(rateOfFire)
