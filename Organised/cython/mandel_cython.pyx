# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:56:53 2021

@author: claus
"""
import cython
import numpy as np
cimport numpy as np

ctypedef np.complex128_t cpl_t
cpl = np.complex128


# def mandelbrot_naive_cython(c_grid, MAX_ITER, THRESHOLD):
#     #MAX_ITER = 80
#     #THRESHOLD = 30
#     rows = len(c_grid)
#     cols = len(c_grid[0])
#     diverge_time = np.zeros((rows,cols))
#     for row in range(rows):
#         for col in range(cols):
#             x = c_grid[row,col].real
#             y = c_grid[row,col].imag
#             old_x = x
#             old_y = y
#             for i in range(MAX_ITER):
#                 a = x*x - y*y
#                 b = 2*x*y
#                 x = a+old_x
#                 y = b+old_y
#                 if x*x+y+y > THRESHOLD:
#                     break
#             diverge_time[row][col] = i
#     return diverge_time

def mandelbrot_naive_cython(np.ndarray[cpl_t,ndim=2] c_grid, int MAX_ITER, int THRESHOLD):
    
    cdef int rows = len(c_grid)
    cdef int cols = len(c_grid[0])
    cdef np.ndarray[int, ndim=2] diverge_time = np.zeros((rows,cols), dtype=int)
    
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

def mandelbrot_vectorized_cython(c_grid, MAX_ITER, THRESHOLD):

    cdef int rows = len(c_grid)
    cdef int cols = len(c_grid[0])
    cdef np.ndarray[cpl_t, ndim=2] Z = np.zeros((rows,cols), dtype=cpl)
    
    #cdef np.ndarray[int, ndim=2] M = np.ones((rows,cols), dtype = int)
    #M = np.full((rows, cols), True, dtype=bool)
    #M = np.full((rows, cols), True, dtype=bool)
    cdef np.ndarray[np.uint8_t, ndim = 2, cast=True] M = np.full((rows,cols), True, dtype=bool)
    cdef np.ndarray[int, ndim=2] diverge_time = np.ones((rows,cols), dtype=int)

    for i in range(MAX_ITER):
        Z[M] = Z[M]**2 + c_grid[M]             # Performs the multiplication as a matrix multiplication and adds the corresponding C term to each entry
        diverged = np.greater(np.abs(Z),THRESHOLD,out=np.full(c_grid.shape,0), where=M)   # Finds which entries has diverged beyond threshold
        diverge_time[diverged] = i        # If an entry has diverged, its iteration count is saved
        M[np.abs(Z) > THRESHOLD] = False  # The bool array M keeping track of which entries has diverged is updated
    return diverge_time