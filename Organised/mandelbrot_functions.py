# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:07:53 2021

@author: claus
"""

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time
from dask.distributed import Client, wait, LocalCluster
import multiprocessing as mp
from numba import jit


def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array(
        [np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array(
        [np.linspace(-1.5, 1.5, IM_points), ] * IM_points
        ).transpose()
    return np.complex64(Real + Imag * 1j)


def plot_mandel(fractals):
    plt.imshow(np.log(fractals), cmap=plt.cm.hot, extent=[-2, 1, -1.5, 1.5])


def mandel_gpu(C_host, result_host):

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    C_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C_host)
    result_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, result_host.nbytes)

    prg = cl.Program(ctx,
        """
        #include <pyopencl-complex.h>
        __kernel void mandelbrot_gpu(
            
            __global const cfloat_t *C_gpu,
            __global       int *result_gpu)
            
           {
               int gidx = get_global_id(0);
               int gidy = get_global_id(1);
               int width = get_global_size(0);
               
               
               int i = 0;
               cfloat_t Z = cfloat_new(0,0);
               
            while(cfloat_abs(Z) < 30 && i < 80){
                    Z = cfloat_add(cfloat_mul(Z,Z),C_gpu[gidx*width + gidy]);
                    i++;
                    result_gpu[gidx*width+gidy] = i;
                    }
            }
        """).build()
        
        
    prg.mandelbrot_gpu(queue, C_host.shape,None,C_gpu,result_gpu)
    cl.enqueue_copy(queue, result_host, result_gpu)
    return result_host

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

@jit(nopython=True, parallel=True)
def mandelbrot_naive_numba(c_grid, MAX_ITER, THRESHOLD):
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