# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:39:27 2021

@author: claus
"""
import time
import mandel_cython as mc
import mandel_functions as mf

RE_points = 1000
IM_points = 1000

grid = mf.create_datapoints(RE_points,IM_points)
# start = time.time()
# result = mc.mandelbrot_naive_cython(grid,80,30)
# stop = time.time()

# print(stop-start)
# mf.plot_mandel(result)

start = time.time()
result = mc.mandelbrot_vectorized_cython(grid,80,30)
stop = time.time()

print(stop-start)
mf.plot_mandel(result)