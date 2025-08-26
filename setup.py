from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="coralearn",
    version="0.1.8",
    author="Coralap",
    description="An AI library written using only NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Coralap/CoraLearn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_required=[
        "numpy>=1.23",
    ],
)
