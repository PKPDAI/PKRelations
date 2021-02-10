from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="pk_rex",
    packages=find_packages(),
    install_requires=requirements
)
