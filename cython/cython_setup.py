# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:57:16 2021

@author: claus
"""

from setuptools import setup
from Cython.Build import cythonize

import numpy 

setup(
    ext_modules = cythonize("mandel_cython.pyx"),
    include_dirs=[numpy.get_include()]
)