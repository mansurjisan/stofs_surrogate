"""Setup script for stofs-surrogate package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stofs-surrogate",
    version="0.1.0",
    author="Mansur Jisan",
    author_email="mansur.jisan@noaa.gov",
    description="Physics-Informed GNN Surrogate for STOFS Storm Surge Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mansurjisan/stofs_surrogate",
    packages=find_packages(include=["stofs_surrogate", "stofs_surrogate.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "netCDF4>=1.6.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "cartopy>=0.21.0",
            "cmocean>=3.0.0",
        ],
        "obs": [
            "searvey>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stofs-train=scripts.train_cwl_gnn_optimized_v3:main",
        ],
    },
)
