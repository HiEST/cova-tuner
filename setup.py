#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup module."""

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()


if __name__ == '__main__':
    setup(
        name='edge_autotune',
        version='0.0.1-dev',
        license='Apache-2.0',
        author='Daniel Rivas',
        author_email='daniel.rivas@bsc.es',
        url='https://github.com/HiEST/edgeautotuner',
        py_modules=[
            'edge_autotune',
            'edge_autotune.apps',
            'edge_autotune.utils',
            'edge_autotune.dnn',
        ],
        package_dir={'': 'src'},
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: Unix"
        ],
        description='Edge AutoTune. Framework for automated fine-tuning of edge models',
        long_description=long_description,
        long_description_content_type="text/markdown",
  )
