# setup.py
from setuptools import setup, find_packages

setup(
    name='ouroboros',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.26',
        'scipy>=1.13',
        'zstandard>=0.22',
        'rich>=13.7',
        'pyyaml>=6.0',
        'matplotlib>=3.9',
    ],
    description='Self-bootstrapping mathematical society via MDL compression',
)