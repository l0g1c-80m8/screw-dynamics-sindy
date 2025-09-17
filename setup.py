#!/usr/bin/env python3
"""Setup script for Screw Dynamics SINDy package."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from __init__.py or set manually
__version__ = "1.0.0"

setup(
    name="screw-dynamics-sindy",
    version=__version__,
    author="Screw Dynamics SINDy Contributors",
    author_email="contact@example.com",  # Update with actual contact
    description="Data-driven modeling of robotic screw-driving dynamics using SINDy framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/l0g1c-80m8/screw-dynamics-sindy",
    project_urls={
        "Bug Tracker": "https://github.com/l0g1c-80m8/screw-dynamics-sindy/issues",
        "Documentation": "https://github.com/l0g1c-80m8/screw-dynamics-sindy#readme",
        "Source Code": "https://github.com/l0g1c-80m8/screw-dynamics-sindy",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
        "ros": [
            # ROS dependencies - install separately via ROS
        ],
    },
    entry_points={
        "console_scripts": [
            "sindy-train=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "data": ["*.csv", "*.png"],
    },
    keywords=[
        "sindy",
        "dynamics",
        "robotics",
        "screw-driving",
        "machine-learning",
        "computer-vision",
        "sparse-identification",
        "pytorch",
    ],
    zip_safe=False,
)