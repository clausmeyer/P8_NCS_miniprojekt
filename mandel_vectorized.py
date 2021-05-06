# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:16:20 2021

@author: claus
"""


import time
import matplotlib.pyplot as mplpp
import numpy as np
import mandel_functions as mf

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
    
RE_points = 1000
IM_points = 1000
    
grid = mf.create_datapoints(RE_points,IM_points)



res = mandelbrot_vectorized(grid,80,30)
mf.plot_mandel(res)