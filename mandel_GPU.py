# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:50:23 2021

@author: claus
"""

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as mplpp

def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return np.complex64(Real + Imag * 1j)

def plot_mandel(fractals):
    ax = mplpp.imshow(np.log(fractals),cmap=mplpp.cm.hot,  extent=[-2,1,-1.5,1.5])
    
    
def mandel_gpu(C_host,result_host):

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
        
    mf = cl.mem_flags
    C_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C_host)
    result_gpu = cl.Buffer(ctx, mf.WRITE_ONLY , result_host.nbytes)
    
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
    
if __name__ == '__main__':
    RE_points = 4000
    IM_points = 4000
    max_its = 80
    grid = create_datapoints(RE_points,IM_points)
    M = np.zeros((grid.shape),dtype = int);
    Z = grid
    res = mandel_gpu(grid,M)
    plot_mandel(res)