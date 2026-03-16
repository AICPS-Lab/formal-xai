"""formal-xai — Formally Verified Explainable AI."""

from setuptools import setup, find_packages

setup(
    name="formal-xai",
    version="0.1.0",
    description="VitaX: Formally verified attribution explanations for neural networks",
    author="VitaX Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9",
        "numpy>=1.20",
        "matplotlib>=3.3",
        "scikit-learn>=0.24",
        "tqdm>=4.0",
        "pyyaml>=5.0",
    ],
    extras_require={
        "captum": ["captum>=0.5"],
        "nnv": ["matlabengine"],
        "marabou": ["maraboupy"],
        "colorama": ["colorama>=0.4"],
        "dev": ["pytest>=7.0", "pandas"],
        "all": ["captum>=0.5", "colorama>=0.4", "pandas"],
    },
)
