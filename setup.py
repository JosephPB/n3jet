# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
import subprocess
import os
from os.path import abspath, dirname, join
from glob import glob

this_dir = abspath(dirname(__file__))
#with open(join(this_dir, "LICENSE")) as f:
#    license = f.read()
    
with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

#scripts = glob("scripts/*.py") + glob("scripts/*.sh")

setup(
        name="n3jet",
        version="1.0",
        description="",
        url="https://github.com/JosephPB/n3jet/",
        long_description_content_type='text/markdown',
        long_description=long_description,
        scripts=scripts,
        author="Joseph Bullock",
        author_email='j.p.bullock@durhan.ac.uk',
        #license="MIT license",
        install_requires=requirements,
        packages = find_packages(exclude=["docs"]),
        py_modules=[splitext(basename(path))[0] for path in glob('n3jet/*.py')],
        include_package_data=True
)

