#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:23:52 2021

@author: ksxuphy
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import csv



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

dt =0.01


def spiketimes_per(spikestrain,ind):
    spikes = spikestrain
    with open("EXSpikes_per_%s.txt"%ind,'w') as f:
        writer=csv.writer(f,'excel')
        flag = 0
        for sp in spikes:
            if flag <150.and flag >=0 :
                writer.writerow(['%.3f'%(x*dt) for x in sp])
            flag =flag+1
            
def spiketimes_nonper(spikestrain,ind):
    spikes = spikestrain
    with open("EXSpikes_nonper_%s.txt"%ind,'w') as f:
        writer=csv.writer(f,'excel')
        flag = 0
        for sp in spikes:
            if flag <500.and flag >=150 :
                writer.writerow(['%.3f'%(x*dt) for x in sp])
            flag =flag+1
            
            
def spiketimes_inhi(spikestrain,ind):
    spikes = spikestrain
    with open("EXSpikes_inhi_%s.txt"%ind,'w') as f:
        writer=csv.writer(f,'excel')
        flag = 0
        for sp in spikes:
            if flag <1000.and flag >=500 :
                writer.writerow(['%.3f'%(x*dt) for x in sp])
            flag =flag+1

'导入spikes数据'
spikes_1 = read_and_fill('spikes_1_1.5.txt')
spikes_2 = read_and_fill('spikes_4_1.5.txt')
spikes_3 = read_and_fill('spikes_1_2.5.txt')
spikes_4 = read_and_fill('spikes_4_2.5.txt')




spiketimes_per(spikes_3,'_1_2.5')
spiketimes_nonper(spikes_3,'_1_2.5')
spiketimes_inhi(spikes_3,'_1_2.5')







