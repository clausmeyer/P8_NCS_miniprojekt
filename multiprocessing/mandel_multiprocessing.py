# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:40:13 2021

@author: claus
"""


import time
import multiprocessing as mp
import matplotlib.pyplot as mplpp
import numpy as np

def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return Real + Imag * 1j  

def plot_mandel(fractals):
    ax = mplpp.imshow(np.log(fractals),cmap=mplpp.cm.hot,  extent=[-2,1,-1.5,1.5])

def mandelbrot_parallel(full_grid, MAX_ITER, THRESHOLD, processes,i):
    
    c_grid = full_grid[i]
    
    rows = len(c_grid)
    cols = len(c_grid[0])
     
    Z = np.zeros((rows, cols), dtype=np.complex128)
    M = np.full((rows, cols), True, dtype=bool)
    diverge_time = np.zeros(Z.shape, dtype=int)

    for i in range(MAX_ITER):
        Z[M] = Z[M]**2 + c_grid[M]             # Performs the multiplication as a matrix multiplication and adds the corresponding C term to each entry
        diverged = np.greater(np.abs(Z),THRESHOLD,out=np.full(c_grid.shape,False), where=M)   # Finds which entries has diverged beyond threshold
        diverge_time[diverged] = i        # If an entry has diverged, its iteration count is saved
        M[np.abs(Z) > THRESHOLD] = False  # The bool array M keeping track of which entries has diverged is updated
    return diverge_time
    

def parallel_mandelbrot(P,N,L):
    pool = mp.Pool(processes=P)
    
    
    results = [pool.apply_async(mandelbrot_parallel,(para_grid,80,30,P,i)) for i in range(N)]
    
    pool.close()
    pool.join()
    K_values = [result.get() for result in results]
    return K_values

def plot_mandel(fractals):      
    mplpp.imshow(np.log(fractals),cmap=mplpp.cm.hot)    # Plots the mandelbrot set


if __name__ == '__main__':
    P = 8 # number of processors
    N = P
    L = 100
    RE_points = 2000
    IM_points = 2000
    grid = create_datapoints(RE_points,IM_points)
    para_grid = np.zeros((P,int(RE_points),int(IM_points/(P))), dtype = np.complex128)
    for i in range(int(P)):
        para_grid[i,:,:] = grid[0:int(IM_points),int(RE_points/(P)*(i)):int(RE_points/(P)*(i+1))]
    start = time.time()
    results = parallel_mandelbrot(P,N,L)
    stop = time.time()
    print(stop-start)
   
    
    test = np.hstack([row for row in results])
        
    plot_mandel(test)