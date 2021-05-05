# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:55:02 2021

@author: claus
"""
import numpy as np

def create_datapoints(RE_points: int, IM_points: int):

    Real = np.array([np.linspace(-2, 1, RE_points), ] * RE_points)
    Imag = np.array([np.linspace(-1.5, 1.5, IM_points), ] * IM_points).transpose()
    return Real + Imag * 1j     

