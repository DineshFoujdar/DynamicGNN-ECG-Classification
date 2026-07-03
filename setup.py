"""
Setup file for TGLLNet-ECG-Classification
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tgllnet-ecg-classification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Task-Guided Lead Correlation Learning for Multi-label ECG Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TGLLNet-ECG-Classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "wfdb>=4.0.0",
        "PyWavelets>=1.1.0",
        "tqdm>=4.60.0",
        "jupyter>=1.0.0",
        "scipy>=1.5.0",
    ],
)