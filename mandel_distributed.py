# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:49:59 2021

@author: claus
"""
#import mandel_vectorized as mv
import time
from dask.distributed import Client, wait, LocalCluster
import matplotlib.pyplot as mplpp
import numpy as np

#import mandel_functions as mf

def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return Real + Imag * 1j     

def plot_mandel(fractals):
    ax = mplpp.imshow(np.log(fractals),cmap=mplpp.cm.hot,  extent=[-2,1,-1.5,1.5])
    
def mandelbrot_vectorized(c_grid, MAX_ITER, THRESHOLD):

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

def parallel_mandel(P,grid):
    
    with LocalCluster(
            n_workers=P,
            #threads_per_worker=1,
            #memory_limit='2GB',
            #ip='tcp://localhost:9895',
    ) as cluster, Client(cluster) as client:
        
        # Do something using 'client'   
        results = []
    
        
        for block in range(P):
            para_grid = grid[0:int(IM_points),int(RE_points/(P)*(block)):int(RE_points/(P)*(block+1))]
            results.append(
                client.submit(
                    mandelbrot_vectorized,para_grid,80,30
                    )
                )

        wait(results)            
        res = client.gather(results)
        return res

if __name__ == '__main__':
    P = 4 # number of processors
    N = P
    L = 100
    RE_points = 1000
    IM_points = 1000

    grid = create_datapoints(RE_points,IM_points)
    #para_grid = np.zeros((P,int(RE_points/2),int(IM_points/(P/2))), dtype = np.complex128)
    res = parallel_mandel(P,grid)
    full_grid = np.hstack([row for row in res])
    plot_mandel(full_grid)