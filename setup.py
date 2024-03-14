#!/usr/bin/env python

from setuptools import setup

setup(
    name="Multi-Plane Light Conversion",
    url="https://github.com/joaocarloscabreu/MPLC",
    author="Joao Abreu",
    author_email="joaocarloscabreu@gmail.com",
    packages=["MPLC"],
    install_requires=["numpy","matplotlib","scipy"],
    version="1.0",
    license="CC BY 3.0",
    description="A python code simulating the Multi-Plane Light Conversion technology based on the article https://doi.org/10.1038/s41467-019-09840-4"
)