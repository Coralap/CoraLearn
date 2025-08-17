from setuptools import setup, find_packages

setup(
    name="coralearn",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23"  # any dependencies your library needs
    ],
    author="Coralap",
    description="An AI library written using only numpy.",
    python_requires='>=3.8',
)
