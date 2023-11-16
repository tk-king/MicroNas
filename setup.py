#!/usr/bin/env python

from setuptools import setup
import pathlib
import os


root = pathlib.Path(__file__).parent
os.chdir(str(root))


setup(name='MicroNAS',
      version='1.0.0',
      description='Hardware aware neural architecture search for time series classification',
      author='Tobias King',
      author_email='tk.king.dev@gmail.com',
      packages=['micronas']
     )