# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:58:45 2021

@author: claus
"""

import time

import matplotlib.pyplot as mplpp
import numpy as np
#import mandel_functions as mf
import multiprocessing as mp


def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return Real + Imag * 1j     

def plot_mandel(fractals):
    ax = mplpp.imshow(np.log(fractals),cmap=mplpp.cm.hot,  extent=[-2,1,-1.5,1.5])
    
def mandelbrot_parallel(c_grid, MAX_ITER, THRESHOLD, processes):

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

def parallel_setup(P,N):
    pool = mp.Pool(processes=P)


    res = [pool.apply_async(mandelbrot_parallel, (para_grid[i,:,:],80,30,P)) for i in range(N)]

    pool.close()
    pool.join()
    #K_values = [res.get() for result in res]
    return res
    
if __name__ == '__main__':
    P = 16 # number of processors
    N = P
    L = 100
    RE_points = 1000
    IM_points = 1000

    grid = create_datapoints(RE_points,IM_points)
    para_grid = np.zeros((P,int(RE_points/2),int(IM_points/(P/2))), dtype = np.complex128)
    for i in range(int(P/2)):
        para_grid[i,:,:] = grid[0:int(1000/2),0:int(1000/(P/2))]
         #para_grid = grid[0:int(1000/2),0:int(1000/(P/2))]
    # res = mandelbrot_vectorized(para_grid,80,30)
    # mf.plot_mandel(res)

    res = parallel_setup(P,N)
   # plot_mandel(res)