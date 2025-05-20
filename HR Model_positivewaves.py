"""
Created Runzhou liu and Kesheng Xu
ksxu@ujs.edu.cn or keshengxuu@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

# change the linewidth of the axes and spines
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["lines.linewidth"] = 4
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 2
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["ytick.minor.width"] = 2
# change the fontsize of the ticks label
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
# change the fontsize of the axes label
plt.rcParams["axes.labelsize"] = 20
# change the fontsize of the legend
plt.rcParams["legend.fontsize"] = 20
# change the fontsize of the title
plt.rcParams["axes.titlesize"] = 20
# change the title font size
plt.rcParams["font.size"] = 20
# change the font family to Arial
plt.rcParams["font.family"] = "DejaVu Sans"

'flags'
make_weights = 1;
simulate_network = 1;

plot_activity_samples = 1;

'parameters for simuliation and neural networks'
dt = 0.01; # time step 
tau = 10; # time constants for population models

NE = 500; # the number of excitatory neurons
NI = 500; # the number of inhibitory neurons
N = NE+NI;

weight_spec = 1; # the synaptic weights
conn_spar_EE = 1; #the probality of the connections

m = 0.5  # specific connectivity

mEE = m
mEI = m
mIE = m
mII = m

po_exc = np.linspace(0, np.pi, NE)
po_inh = np.linspace(0, np.pi, NI)
po_all = np.concatenate([po_exc, po_inh])

pert_size = 0.1

ts_dur = [100] # duration of simulation


# the number of perturbation neurons
R = 150
Ns_pert = [int(R)] 



# Hindmarsh-Rose neural network
def simulate_dynamics(I, N, T, w, dt, tau):
    
    x_i = np.zeros((N, len(T) + 1))
    y_i = np.zeros((N, len(T) + 1))
    z_i = np.zeros((N, len(T) + 1))
    
    x_i = np.random.uniform(-2,1,size=x_i.shape)
    y_i = np.random.uniform(-2,1,size=x_i.shape)
    z_i = np.random.uniform(-3,-2,size=x_i.shape)
    
    r = np.zeros((N, len(T) + 1))
    
    a=1.0
    b=3.0
    c=1.0
    d=5.0
    e=4.0
    f=0.006
    g=-1.6
    i_ext=1.5
    
    for i in range(len(T)):
        
        inp = np.dot(w.T, r[:, i]) + I[:,i]
        
        x_i[:, i+1] = x_i[:, i] + dt * ( y_i[:, i]  - a* x_i[:, i]**3 + b* x_i[:, i]**2 + i_ext - z_i[:, i] + inp)
        y_i[:, i+1] = y_i[:, i] + dt * ( c - d*x_i[: ,i]**2 - y_i[:,i])
        z_i[:, i+1] = z_i[:, i] + dt * (f * (e * (x_i[:, i] - g) - z_i[:, i]))
        
        r[:, i+1] = r[:, i] + dt/tau * (-r[:, i] + NL(inp))

    v = x_i
    
    return r, v



def NL(zi):

    zo = np.zeros_like(zi)
    zo[zi >= 0] = zi[zi >= 0]
    zo[zi < 0] = 0

    return zo



# half-wave rectification
def rectify(z):
    zr = z * (z > 0)
    return zr



# adding perturbation to the neural network
def pert_sim(N, NE, w, dt, tau, pert_ids, pert_size, t_dur, t_betw, learning_rule='covariance'):

    t0 = 1600
    
    N_pert = len(pert_ids)
    
    N_rep = 10
    t_dur_tot = (t_dur+t_betw)*N_rep
    
    t_end = t0 + t_dur_tot + t0
    T = np.arange(0, t_end, dt)
    
    # Generate perturbation input with random amplitude
    I_genPert = 1.0 + 0.1*(np.random.rand(N, len(T)))
    dI_pert = np.zeros((N, len(T)))
    
    #  perturbation of  positive square waves
    t_ids = []
    
    # Generate time indices for perturbation application
    for i in range(1,N_rep+1):  
        t1 = t0 + (t_dur+t_betw)*(i-1)
        t2 = t1 + t_dur
        t_ids = t_ids + list(np.where((T >= t1) & (T <= t2))[0])
    
    # Apply perturbation to specific neurons at specific times
    for i in range(1,N_pert+1):
        dI_pert[pert_ids[i-1], t_ids] = pert_size
    
    # Simulate dynamics of the network with perturbation
    # r, v= simulate_dynamics(I_genPert, N, T, w, dt, tau)
    r, v= simulate_dynamics(I_genPert + dI_pert, N, T, w, dt, tau)
    
    return v, r



# the degree of synatic coupling regulated by parameter K
k = 6

J0 = 1 / NE * 2 # 0.004
JEE = J0
JEI = J0 * k
JIE = -J0 * k 
JII = -J0 * k 


if make_weights:
    # Initialize weight matrices
    wEE = np.zeros((NE, NE))
    wEI = np.zeros((NE, NI))
    wIE = np.zeros((NI, NE))
    wII = np.zeros((NI, NI))
    
    # Compute weights for excitatory to excitatory connections
    for i in range(NE):
        wEE[i, :] = (1 + mEE * np.cos(2 * (po_exc[i] - po_exc))) * JEE

    # Compute weights for excitatory to inhibitory connections
    for i in range(NE):
        wEI[i, :] = (1 + mEI * np.cos(2 * (po_exc[i] - po_inh))) * JEI
    
    # Compute weights for inhibitory to excitatory connections
    for i in range(NI):
        wIE[i, :] = (1 + mIE * np.cos(2 * (po_inh[i] - po_exc))) * JIE
    
    # Compute weights for inhibitory to inhibitory connections
    for i in range(NI):
        wII[i, :] = (1 + mII * np.cos(2 * (po_inh[i] - po_inh))) * JII
    
    # Apply connectivity sparsity to excitatory to excitatory connections
    wEE = wEE * np.random.binomial(1, conn_spar_EE, (NE, NE))

    # Combine weight matrices into a single matrix
    w = np.block([[wEE, wEI], [wIE, wII]])
    
    # Add random perturbation to all weights
    w = w * (1 + 1/2 * (np.random.rand(N, N) - 0.5))
    
    # Apply rectification to the weights
    w[:NE, :] = rectify(w[:NE, :])
    w[NE:, :] = -rectify(-w[NE:, :])
    
    # Zero out the diagonal elements of the weight matrix
    np.fill_diagonal(w, 0)   
        
        
        
# simulate
if simulate_network:

        r_all = []
        v_all = []
        dw_all = []
        pert_ids_all = []

        avg_ind = np.zeros((len(Ns_pert), len(ts_dur))) 
        avg_ind_all = np.zeros_like(avg_ind) 
        tot_ind = np.zeros((len(Ns_pert), len(ts_dur))) 
        tot_ind_all = np.zeros_like(tot_ind)
        avg_ind_w = np.zeros((len(Ns_pert), len(ts_dur)))

        for k1 in range(len(Ns_pert)):  #looping for number of pertutbion neurons
            N_pert = Ns_pert[k1]

            for k2 in range(len(ts_dur)): #looping for the duration time of the pertutbion neurons
                t_dur = ts_dur[k2]
                t_betw = t_dur

                  # perturbed neurons
             
                iz = 1
                pert_ids = np.arange(1, NE+1)[iz-1:iz-1+N_pert]
                pert_ids = np.squeeze(pert_ids)  
                pert_ids_all.append(pert_ids)   # listing or saving the index pert_ids for N_pert =50,100,150,200,respectively.    
                           
                [v, r] = pert_sim(N, NE, w, dt, tau, pert_ids, pert_size, t_dur, t_betw)



O = 20000   # move the transient data 



# detect the spikes
theta = 0
v_all = v[:, O:]
spikes_All = [(np.diff(1 * (v_all[i, :] > theta)) == 1).nonzero()[0] for i in range(1000)]

#save spikes to the datafile
with open('spikes_1.5P.txt', 'w') as file:
    for row in spikes_All:
        line = ' '.join(map(str, row))
        file.write(line + '\n')



# Create a figure and axis for the combined plot
fig, ax1 = plt.subplots(figsize=(20, 6))

# Scatter plot
for i, matrix in enumerate(spikes_All):
    if i < R:
        color = 'red'
    elif R <= i < 500:
        color = 'blue'
    else:
        color = 'green'

    x_values = matrix.flatten()
    y_values = [i] * len(x_values)
    ax1.scatter(x_values*dt, y_values, color=color, alpha=0.5, s=0.5)
    
    plt.xlabel('Times [ms]')
    plt.ylabel('Neuron index')

plt.xlim([0,5000])
plt.ylim([0,1000])
plt.tight_layout()
plt.show()
plt.savefig('positive_square_wave.png',dpi =300)

# firing rate
y = r[:NE, O:]       

#save data
avg = np.mean(y[:150, :], axis=0)
avg = avg[::5]
avg = ' '.join(map(str, avg))

with open(f"y1_1.txt", 'w') as file:
    file.write(avg + '\n')
    
avg1 = np.mean(y[160:215, :], axis=0)
avg1 = avg1[::5]
avg1 = ' '.join(map(str, avg1))

with open(f"y1_2.txt", 'w') as file:
    file.write(avg1 + '\n')
    
avg2 = np.mean(y[215:270, :], axis=0)
avg2 = avg2[::5]
avg2 = ' '.join(map(str, avg2))
with open(f"y1_3.txt", 'w') as file:
    file.write(avg2 + '\n')


    


            


        
    




