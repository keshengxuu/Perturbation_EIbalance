import numpy as np

def ISI_Phase(Num_Neurs, TIME, spikes, start_neuron=151, end_neuron=500):
    Phase = np.zeros((TIME.size, Num_Neurs), dtype=np.float32)   
    for ci in range(start_neuron-1, end_neuron):  #  looping the neurons
        mth = 0  # index for recording spikes
        nt = 0   # count spike times
        for t in TIME:    
            if np.sum(spikes[ci]) < 2:   
                Phase[nt, ci] = 0
            elif (mth+1) < np.array(spikes[ci]).size:  
                # 计算相位
                Phase[nt, ci] = 2.0 * np.pi * (t - spikes[ci][mth]) / (spikes[ci][mth+1] - spikes[ci][mth])
                nt = nt + 1
                if t > spikes[ci][mth+1]:   
                    mth = mth + 1
    return Phase

#loading the spike times for calculating the phasea
dt = 0.01   
spikes = []
with open('spikes_3.0.txt', 'r') as f:
    for line in f:
        spike_times = [float(x) * dt for x in line.strip().split()]
        spikes.append(np.array(spike_times))


Total = 5000
TIME = np.arange(0, Total, dt)
Num_Neurs = 1000


Phase = ISI_Phase(Num_Neurs, TIME, spikes)


Phase_cut_1 = Phase[:140000, 150:500]
Phase_cut_2 = Phase[140000:340000, 150:500]
Phase_cut_3 = Phase[340000:500000, 150:500]

np.savetxt("phase_3.0_1.txt", Phase_cut_1, fmt='%.3f', delimiter='\t')
np.savetxt("phase_3.0_2.txt", Phase_cut_2, fmt='%.3f', delimiter='\t')
np.savetxt("phase_3.0_3.txt", Phase_cut_3, fmt='%.3f', delimiter='\t')