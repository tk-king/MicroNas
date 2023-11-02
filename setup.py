#!/usr/bin/env python

from distutils.core import setup


setup(name='MicroNAS',
      version='1.0.0',
      description='Hardware aware neural architecture search for time series classification',
      author='Tobias King',
      author_email='tk.king.dev@gmail.com',
      packages=['micronas'],
      package_dir={"micronas": "src"}
     )