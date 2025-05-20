import numpy as np
import matplotlib.pyplot as plt


nnodes = 350  # number of neurons
runTime = 1600  #  simulation times
runInt = 0.01  # time steps
Trun = np.arange(0, runTime, runInt)   

# load the data
phase_data = np.loadtxt('phase_3.0_3.txt', delimiter='\t') 


assert phase_data.shape == (160000, 350), f"Expected shape (140000, 350), but got {phase_data.shape}"


sampling_rate = 10  
sampled_indices = np.arange(0, phase_data.shape[0], sampling_rate)  
phaseMaxF = phase_data[sampled_indices, :] 

print(f"Sampled data shape: {phaseMaxF.shape}")


Pcoher = np.zeros((nnodes, nnodes))
for ii in range(nnodes):
    for jj in range(ii):  
        Pcoher[ii, jj] = np.abs(np.mean(np.exp(1j * (phaseMaxF[:, ii] - phaseMaxF[:, jj]))))


PcoherMean = np.mean(Pcoher[np.tril_indices(nnodes, k=-1)])  
PcoherVar = np.var(Pcoher[np.tril_indices(nnodes, k=-1)])  


np.savetxt("PLV_3.0_3.txt", Pcoher, fmt='%.6g', delimiter='\t')


plt.figure(figsize=(8, 6))
plt.imshow(Pcoher + Pcoher.T + np.eye(nnodes), cmap='jet', vmin=0.4, vmax=1, interpolation='none')
plt.colorbar(label='Phase Locking Value')
plt.title(f'Phase Locking Matrix (Pcoher) - Sampled (every {sampling_rate} points)')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')

plt.text(0.05, 0.9, f'Mean: {PcoherMean:.4f}\nVar: {PcoherVar:.4f}',
         transform=plt.gca().transAxes, fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig('Figure_3.0_3.png', dpi=300, bbox_inches='tight')
plt.show()

