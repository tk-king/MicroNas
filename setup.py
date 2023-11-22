#!/usr/bin/env python

from setuptools import setup, find_packages
import pathlib
import os

root = pathlib.Path(__file__).parent
os.chdir(str(root))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='MicroNAS',
    version='1.0.0',
    description='Hardware-aware neural architecture search for time series classification',
    author='Tobias King',
    author_email='tk.king.dev@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8'
)
