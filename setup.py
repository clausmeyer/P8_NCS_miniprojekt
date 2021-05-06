# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:57:16 2021

@author: claus
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("helloworld.pyx")
)