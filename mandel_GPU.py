# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:50:23 2021

@author: claus
"""

import pyopencl as cl
import numpy as np

def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return np.complex64(Real + Imag * 1j)

def mandel_gpu(Z_host,C_host,M_host):

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
        
    mf = cl.mem_flags
    C_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C_host)
    Z_gpu = cl.Buffer(ctx, mf.READ_WRITE , Z_host.size)
    
    #MAX_ITERS = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iters)
    #result_gpu = cl.Buffer(ctx, mf.WRITE_ONLY,result_host.nbytes)
    M_gpu = cl.Buffer(ctx, mf.READ_WRITE , M_host.size)
    prg = cl.Program(ctx, 
        """
        #include <pyopencl-complex.h>
        __kernel void mandelbrot_gpu(
            
            __global const cfloat_t *C_gpu,
            __global       cfloat_t *Z_gpu,
            __global       int *M_gpu)
            
           {
               int gid = get_global_id(0);
               int i = 0;
               cfloat_t Z = cfloat_new(0,0);
            while(cfloat_abs(Z) < 30 && i < 80){
                    Z = cfloat_add(cfloat_mul(Z,Z),C_gpu[gid]);
                    i++;
                    M_gpu[gid] = i;
                    }
            }
        """).build()
        
        
    prg.mandelbrot_gpu(queue, Z_host.shape[0],None,C_gpu,Z_gpu,M_gpu)
    cl.enqueue_copy(queue, M_host, M_gpu)

    
if __name__ == '__main__':
    RE_points = 2000
    IM_points = 2000
    max_its = 80
    grid = create_datapoints(RE_points,IM_points)
    M = np.zeros(grid.shape);
    Z = grid
    mandel_gpu(Z,grid,M)