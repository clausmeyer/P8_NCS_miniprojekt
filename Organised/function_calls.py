# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:10:48 2021

@author: claus
"""
import numpy as np
import mandelbrot_functions as mf
import time 

# ----------------------------GPU version-----------------------------------
# if __name__ == '__main__':
#     RE_points = 4000
#     IM_points = 4000
#     max_its = 80
#     grid = mf.create_datapoints(RE_points,IM_points)
#     M = np.zeros((grid.shape),dtype = int);
#     Z = grid
#     res = mf.mandel_gpu(grid,M)
#     mf.plot_mandel(res)



# ----------------------------Vectorised version-----------------------------------
# if __name__ == '__main__':
#     RE_points = 1000
#     IM_points = 1000
    
#     grid = mf.create_datapoints(RE_points,IM_points)
#     res = mf.mandelbrot_vectorized(grid,80,30)
#     mf.plot_mandel(res)

# ----------------------------Naive version-----------------------------------
# if __name__ == '__main__':
#     RE_points = 1000
#     IM_points = 1000
    
#     grid = mf.create_datapoints(RE_points,IM_points)
#     start_time = time.time()
#     res = mf.mandelbrot_naive(grid,80,30)
#     stop_time = time.time()
#     print(stop_time-start_time)
#     mf.plot_mandel(res)

# ----------------------------Numba version-----------------------------------
if __name__ == '__main__':  
    RE_points = 1000
    IM_points = 1000
    
    grid = mf.create_datapoints(RE_points,IM_points)
    
    start_time = time.time()
    res = mf.mandelbrot_naive_numba(grid,80,30)
    stop_time = time.time()
    print(stop_time-start_time)
    mf.plot_mandel(res)
    
    
    
    mf.plot_mandel(res)
    
