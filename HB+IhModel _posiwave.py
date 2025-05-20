import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

make_weights = 1
simulate_network = 1


dt = 0.01
tau = 10

NE = 500
NI = 500
N = NE + NI

weight_spec = 1
conn_spar_EE = 1

m = 0.5
mEE = m
mEI = m
mIE = m
mII = m

po_exc = np.linspace(0, np.pi, NE)
po_inh = np.linspace(0, np.pi, NI)
po_all = np.concatenate([po_exc, po_inh])

pert_size = 0.1
ts_dur = [100]

R = 150
Ns_pert = [int(R)]



# HyB模型参数
gd, gr, gsr, gl = 2.5, 2.8, 0.28, 0.06

gsd, gh = 0.203, 0.335  # chaotic
# gsd, gh = 0.249, 0.23  #nonchaotic

V0d, V0r, zd, zr, tr = -25, -25, 0.25, 0.25, 2
V0sd, zsd, tsd = -40, 0.11, 10
eta, kappa, tsr = 0.014, 0.18, 35
V0h, zh, th = -85, -0.14, 125
Ed, Er, El, Eh = 50, -90, -80, -30
temp = 36
rho = 1.3**((temp-25.)/10)
phi = 3**((temp-25.)/10)





def simulate_dynamics(I, N, T, w, dt, tau):

    v = np.zeros((N, len(T) + 1))    
    ar = np.zeros((N, len(T) + 1))   
    asd = np.zeros((N, len(T) + 1))  
    ca = np.zeros((N, len(T) + 1))   
    ah = np.zeros((N, len(T) + 1))   
    r = np.zeros((N, len(T) + 1))    

    
    # v[:, 0] = np.random.uniform(-70, -50, N)
    
    
    v[:, 0] = -60
    
    for i in range(N):
        ar[i, 0] = 1 / (1 + np.exp(-zr * (v[i, 0] - V0r)))
        asd[i, 0] = 1 / (1 + np.exp(-zsd * (v[i, 0] - V0sd)))
        ca[i, 0] = -eta * rho * gsd * asd[i, 0] * (v[i, 0] - Ed) / kappa
        ah[i, 0] = 1 / (1 + np.exp(-zh * (v[i, 0] - V0h)))

   
    for i in range(len(T)):
       
        inp = np.dot(w.T, r[:, i]) + I[:, i]

       
        ad = 1 / (1 + np.exp(-zd * (v[:, i] - V0d)))
        isd = rho * gsd * asd[:, i] * (v[:, i] - Ed)
        Imemb = (isd + rho * gd * ad * (v[:, i] - Ed) +
                 rho * (gr * ar[:, i] + gsr * (ca[:, i]**2) / (ca[:, i]**2 + 0.4**2)) * (v[:, i] - Er) +
                 rho * gl * (v[:, i] - El) +
                 rho * gh * ah[:, i] * (v[:, i] - Eh))

       
        v[:, i+1] = v[:, i] + dt * (-Imemb + inp)
        arinf = 1 / (1 + np.exp(-zr * (v[:, i] - V0r)))
        asdinf = 1 / (1 + np.exp(-zsd * (v[:, i] - V0sd)))
        ahinf = 1 / (1 + np.exp(-zh * (v[:, i] - V0h)))
        ar[:, i+1] = ar[:, i] + dt * phi * (arinf - ar[:, i]) / tr
        asd[:, i+1] = asd[:, i] + dt * phi * (asdinf - asd[:, i]) / tsd
        ca[:, i+1] = ca[:, i] + dt * phi * (-eta * isd - kappa * ca[:, i]) / tsr
        ah[:, i+1] = ah[:, i] + dt * phi * (ahinf - ah[:, i]) / th

       
        r[:, i+1] = r[:, i] + dt / tau * (-r[:, i] + NL(inp))

    return r, v



# nonlinear function
def NL(zi):
    zo = np.zeros_like(zi)
    zo[zi >= 0] = zi[zi >= 0]
    return zo




def rectify(z):
    zr = z * (z > 0)
    return zr




def pert_sim(N, NE, w, dt, tau, pert_ids, pert_size, t_dur, t_betw, learning_rule='covariance'):
    t0 = 1600
    N_pert = len(pert_ids)
    N_rep = 10
    t_dur_tot = (t_dur + t_betw) * N_rep
    t_end = t0 + t_dur_tot + t0
    T = np.arange(0, t_end, dt)
    t_ids = []

   
    for i in range(1, N_rep + 1):
        t1 = t0 + (t_dur + t_betw) * (i - 1)
        t2 = t1 + t_dur
        t_ids = t_ids + list(np.where((T >= t1) & (T <= t2))[0])

   
    I_genPert = 1.0 + .1 * (np.random.rand(N, len(T)))
    dI_pert = np.zeros((N, len(T)))
    for i in range(1, N_pert + 1):
        dI_pert[pert_ids[i-1], t_ids] = pert_size

   
    r, v = simulate_dynamics(I_genPert + dI_pert, N, T, w, dt, tau)
    # r, v = simulate_dynamics(I_genPert, N, T, w, dt, tau)
    I = I_genPert + dI_pert
    return v, r, I




k = 6
J0 = 1 / NE * 2
JEE = J0
JEI = J0 * k
JIE = -J0 * k
JII = -J0 * k




if make_weights:
    wEE = np.zeros((NE, NE))
    wEI = np.zeros((NE, NI))
    wIE = np.zeros((NI, NE))
    wII = np.zeros((NI, NI))

    for i in range(NE):
        wEE[i, :] = (1 + mEE * np.cos(2 * (po_exc[i] - po_exc))) * JEE
    for i in range(NE):
        wEI[i, :] = (1 + mEI * np.cos(2 * (po_exc[i] - po_inh))) * JEI
    for i in range(NI):
        wIE[i, :] = (1 + mIE * np.cos(2 * (po_inh[i] - po_exc))) * JIE
    for i in range(NI):
        wII[i, :] = (1 + mII * np.cos(2 * (po_inh[i] - po_inh))) * JII

    wEE = wEE * np.random.binomial(1, conn_spar_EE, (NE, NE))
    w = np.block([[wEE, wEI], [wIE, wII]])
    w = w * (1 + 1/2 * (np.random.rand(N, N) - 0.5))
    w[:NE, :] = rectify(w[:NE, :])
    w[NE:, :] = -rectify(-w[NE:, :])
    np.fill_diagonal(w, 0)




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

    for k1 in range(len(Ns_pert)):
        N_pert = Ns_pert[k1]
        for k2 in range(len(ts_dur)):
            t_dur = ts_dur[k2]
            t_betw = t_dur

          
            iz = 1
            pert_ids = np.arange(1, NE+1)[iz-1:iz-1+N_pert]
            pert_ids = np.squeeze(pert_ids)
            pert_ids_all.append(pert_ids)

            [v, r, I] = pert_sim(N, NE, w, dt, tau, pert_ids, pert_size, t_dur, t_betw)



 
    O = 20000  



   
    theta = -20  
    v_all = v[:, O:]  
    spikes_All = [(np.diff(1 * (v_all[i, :] > theta)) == 1).nonzero()[0] for i in range(N)]

    with open('spikes_NoC_0.1.txt', 'w') as file:
        for row in spikes_All:
            line = ' '.join(map(str, row))
            file.write(line + '\n')



  
    fig, ax1 = plt.subplots(figsize=(15, 6))
    for i, matrix in enumerate(spikes_All):
        if i < R:
            color = 'red'
        elif R <= i < NE:
            color = 'blue'
        else:
            color = 'green'
        x_values = matrix.flatten()/100
        y_values = [i] * len(x_values)
        ax1.scatter(x_values, y_values, color=color, alpha=0.5, s=0.5)
    plt.xlabel('Time[ms]')
    plt.ylabel('Neuron index')
    plt.show()
    
    

A = v[:NE, O:] 

A_downsampled = A[:, ::10]  
np.savetxt('A_NoC_0.1.txt', A_downsampled, fmt='%.3f', delimiter=' ')
    