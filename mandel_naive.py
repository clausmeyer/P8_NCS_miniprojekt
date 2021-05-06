# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:49:32 2021

@author: claus
"""

import time
import matplotlib.pyplot as mplpp
import numpy as np
import mandel_functions as mf

def mandelbrot_naive(c_grid, MAX_ITER, THRESHOLD):
    #MAX_ITER = 80
    #THRESHOLD = 30
    rows = len(c_grid)
    cols = len(c_grid[0])
    diverge_time = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            x = c_grid[row,col].real
            y = c_grid[row,col].imag
            old_x = x
            old_y = y
            for i in range(MAX_ITER):
                a = x*x - y*y
                b = 2*x*y
                x = a+old_x
                y = b+old_y
                if x*x+y+y > THRESHOLD:
                    break
            diverge_time[row][col] = i
    return diverge_time
        
RE_points = 1000
IM_points = 1000

grid = mf.create_datapoints(RE_points,IM_points)
start_time = time.time()
res = mandelbrot_naive(grid,80,30)
stop_time = time.time()
print(stop_time-start_time)
mf.plot_mandel(res)