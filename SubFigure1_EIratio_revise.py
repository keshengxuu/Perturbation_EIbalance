from __future__ import print_function
import pyspike as spk
from pyspike import SpikeTrain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



plt.rcParams['mathtext.sf'] = 'Arial'
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('font',size=6)
plt.rc('font', weight='bold')
plt.rcParams['axes.labelweight'] = 'bold'



def read_and_fill(filename):
    # Read file content
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Convert lines to list of floats
    data = [list(map(float, line.strip().split())) for line in lines]
    
    # Find the maximum row length
    max_length = max(len(row) for row in data)
    
    # Pad rows with zeros
    matrix = [row + [0.0] * (max_length - len(row)) for row in data]
    
    # Convert to numpy array
    matrix_array = np.array(matrix, dtype=float)
    
    return matrix_array



def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0))  # outward by 10 points
#            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        
        
def rasterplot(spikes,Iext, EIra, Pera, ax,Rmin = 150, Rmax = 500):
    for i, matrix in enumerate(spikes):
        if i < Rmin:
            color = 'red'
        elif Rmin <= i < Rmax:
            color = 'blue'
        else:
            color = 'green'

        x_values = matrix.flatten()
        y_values = [i] * len(x_values)
        ax.scatter(x_values*0.01, y_values, color=color, alpha=0.5, s=0.1)
        if ax == ax1:
            ax.annotate('', xy=(1400, 999), xytext=(1400, 1200),
                    arrowprops=dict(facecolor='red', shrink=0.005))
            ax.annotate('', xy=(3400, 999), xytext=(3400, 1200),
                    arrowprops=dict(facecolor='blue', shrink=0.005))
            ax.text(1550,1080,'Pert onset',color ='red')
            ax.text(3550,1100,'Pert off',color ='blue')
        if ax == ax1 or ax == ax2 or ax == ax3 or ax == ax30:
            ax.plot(xx,yy, 'k--' , linewidth=1)
            ax.plot(xx3,yy, 'k--', linewidth=1 )
        
        ax.set_xlim([1000,4000])
        ax.set_ylim([0,999])
        ax.set_xticks([1400,2500,3400],['1400','2500','3400']) 
      
       
            
        ax.set_yticks([249,499,749,999],['250','500','750','1000']) 
        ax.set_ylabel('Neuron index')
        
        if ax == ax30:
            ax.set_xlabel('Times [ms]')
        ax.text(1500,900,r'$\mathsf{I_{ext}=%s,E:I=%s, Pe:Upe =%s}$'%(Iext,EIra,Pera),bbox=dict(facecolor='white', alpha=0.4, edgecolor='white',pad=0),fontdict=dict(fontsize=5, fontweight='bold'))
        
        
def spikesynplot(spiketrain,ax,minval,maxval,flag = 0):
    spike_sync = spk.spike_sync_matrix(spiketrain, interval=(minval, maxval))
    minvalue = 0.4
    maxvalue = 1.0
    if ax == ax7:
        im1 = ax.imshow(spike_sync, vmin =minvalue, vmax = maxvalue,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'),origin = 'lower')
        cax1 = fig.add_axes([0.93, 0.1, 0.009, 0.8]) #the parameter setting for colorbar position
        cbar1=fig.colorbar(im1, cax=cax1,extend='min')
        #cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
        #cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
        cbar1.set_label(r'Spike-Sync',fontsize='medium',labelpad=2)
        cbar1.set_ticks(np.linspace(minvalue,maxvalue,7))
        ##change the appearance of ticks anf tick labbel
        cbar1.ax.tick_params(labelsize='medium')
        cax1.xaxis.set_ticks_position("top")
        #ax.set_xlabel('Neuron index')
    else:
        ax.imshow(spike_sync, vmin =minvalue, vmax = maxvalue,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'),origin = 'lower')
        
    if ax == ax60 or ax == ax90 or ax == ax120:
        ax.set_xlabel('Neuron index')
    if ax == ax7:
        ax.set_title('During Pert',fontsize = 8,pad = 0 ) 
    if ax == ax4:
        ax.set_title('Before Pert',fontsize = 8,pad = 0 ) 
    if ax == ax10:
        ax.set_title('After Pert',fontsize = 8,pad = 0 ) 
      
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    hist, bins = np.histogram(sp_sync, bins=binss,range = (-0.1, 1.1))
    Var = np.var(sp_sync)
    if flag ==0:
        ax.text(110,320,r'$\mathsf{Var}=$'+'%0.3f'%(Var),bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=5, fontweight='bold',color= 'blue')) 
        ax.set_xticks([0,175,350],['150','325','500'])
        ax.set_yticks([0,175,350],['150','325','500'])
    else:
        ax.text(178,500,r'$\mathsf{Var}=$'+'%0.3f'%(Var),bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=5, fontweight='bold',color= 'blue')) 
        ax.set_xticks([0,140,280,420,560],['240','380','520','660','800'])
        ax.set_yticks([0,140,280,420,560],['240','380','520','660','800']) 

if __name__ == "__main__":
    
    yy = np.linspace(0, 999,20)
    xx = np.ones(np.size(yy))*1400
    xx3 = np.ones(np.size(yy))*3400
    
    'loadspiking data'
    spikes1 = read_and_fill('data/spikes_1_1.5.txt')
    spikes2 = read_and_fill('data/spikes_4_1.5.txt')
    spikes3 = read_and_fill('data/spikes_1_2.5.txt')
    spikes4 = read_and_fill('data/spikes_4_2.5.txt')

    
    
    
    # spike_trainsper = [spk.load_spike_trains_from_txt("data/EXSpikes_per%d.txt"%fl,separator=',',edges=(0, 500000)) for fl in range(3)]
    spike_trainsnonper = [spk.load_spike_trains_from_txt("data/EXSpikes_pernonper%d.txt"%fl,separator=',',edges=(0, 500000)) for fl in range(4)]
    
    
    spiketrain1 = [spikes1,spikes2,spikes3,spikes4]
    #spiketrain1 = [spikes4,spikes5,spikes6]
    
    

    
    fig = plt.figure(1, figsize = (7.5,6))
    plt.clf()
    fl = 1
    
    # gs = GridSpec(nrows=3, ncols=3, left=0.07, right=0.45,top=0.945,bottom=0.09,wspace=0.4,hspace=0.25)
    gs1 = GridSpec(nrows=4, ncols=5, left=0.1, right=0.91,top=0.945,bottom=0.09,wspace=0.4,hspace=0.25)
    

    
    ax1 = fig.add_subplot(gs1[0, 0:2])
    ax2 = fig.add_subplot(gs1[1, 0:2])
    ax3 = fig.add_subplot(gs1[2, 0:2])
    ax30 = fig.add_subplot(gs1[3, 0:2])

    
    ax4 = fig.add_subplot(gs1[0, 2])
    ax5 = fig.add_subplot(gs1[1, 2])
    ax6 = fig.add_subplot(gs1[2, 2])
    ax60 = fig.add_subplot(gs1[3, 2])
    
    
    ax7 = fig.add_subplot(gs1[0, 3])
    ax8 = fig.add_subplot(gs1[1, 3])
    ax9 = fig.add_subplot(gs1[2, 3])
    ax90 = fig.add_subplot(gs1[3, 3])
    
    
    ax10 = fig.add_subplot(gs1[0, 4])
    ax11 = fig.add_subplot(gs1[1, 4])
    ax12 = fig.add_subplot(gs1[2, 4])
    ax120 = fig.add_subplot(gs1[3, 4])


    
    
    
    # axx= [ax1,ax2,ax3]#,ax4,ax5,ax6]
    # axxsp= [ax1b,ax2b,ax3b]#,ax4b,ax5b,ax6b]
    axxsp_xtext = [0.2,0.55,0.05]#,0.2,0.40,0.40]
    
    axxx= [ax1,ax2,ax3,ax30]#,ax10],ax11,ax12]
    a2x = [ax4,ax5,ax6,ax60]#,ax10b,ax11b,ax12b]
    a3x = [ax7,ax8,ax9,ax90]
    a4x = [ax10,ax11,ax12,ax120]
    axxxsp_xtext = [0.2,0.10,0.2,0.2]#,0.2,0.2,0.2]
    binss =20
    cmap00=plt.get_cmap('viridis_r')
    
    
    Iext_set = ['1.5','1.5','2.5','2.5']
    EIratio = ['1:1','4:1','1:1','4:1']
    Peratio = ['3:7','3:7','3:7','3:7']
    
    # for ax0, ax,i,ax_xt,Iext0 in zip(axx,axxsp,range(3),axxsp_xtext,Iext_set):
    #     spikesynplot(spike_trainsper[i],ax)
    #     rasterplot(spiketrain0[i],Iext0,ax0)
       
    #spike_sync = spk.spike_sync_matrix(spike_trainsper[2], interval=(2500, 3500))
    Rindex0 = [150,240,150,240]
    Rindex1 = [500,800,500,800]
    

    for ax0,i,EIra,Pera, Iext0 in zip(axxx,range(4),EIratio,Peratio,Iext_set):
    #    spikesynplot(spike_trainsp[i],ax)
        rasterplot(spiketrain1[i], Iext0, EIra, Pera, ax0, Rmin = Rindex0[i], Rmax = Rindex1[i])
    
    #spike_sync = spk.spike_sync_matrix(spike_trainsper[2], interval=(2500, 3500))
    
    flaglist = [0,1,0,1]
    
    for ax,i in zip(a2x,range(4)):
        spikesynplot(spike_trainsnonper[i],ax,minval=0,maxval = 1300, flag = flaglist[i])
    
    for ax,i in zip(a3x,range(4)):
        spikesynplot(spike_trainsnonper[i],ax,minval=1900,maxval = 3400, flag = flaglist[i])
    
    for ax,i in zip(a4x,range(4)):
        spikesynplot(spike_trainsnonper[i],ax,minval=3400,maxval = 3600, flag = flaglist[i])
      
        
       
        
    plt.figtext(0.007,0.94,r'$\mathsf{A}$',fontsize = 'x-large')
    plt.figtext(0.007,0.71,r'$\mathsf{B}$',fontsize = 'x-large')
    plt.figtext(0.007,0.48,r'$\mathsf{C}$',fontsize = 'x-large')
    plt.figtext(0.007,0.26,r'$\mathsf{D}$',fontsize = 'x-large')
    # plt.figtext(0.48,0.956,r'$\mathsf{B_{1}}$',fontsize = 'x-large')
    # plt.figtext(0.48,0.65,r'$\mathsf{B_{2}}$',fontsize = 'x-large')
    # plt.figtext(0.48,0.34,r'$\mathsf{B_{3}}$',fontsize = 'x-large')
    
    #plt.subplots_adjust(bottom=0.1,left=0.09,wspace = 0.2,hspace = 0.2,right=0.93, top=0.985)
    plt.savefig("SubFigure1_EIratio_revise.png",dpi =300)
    #plt.savefig("Figure4.pdf",dpi =300)
    # plt.savefig("Figure5.eps",dpi =300)
    # plt.savefig("Figure5.pdf",dpi =300)