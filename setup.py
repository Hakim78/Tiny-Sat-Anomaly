# =============================================================================
# Tiny-Sat-Anomaly Setup Configuration
# =============================================================================
"""
Package setup for Tiny-Sat-Anomaly.

Install in development mode:
    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8")
install_requires = [
    line.strip()
    for line in requirements.splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="tiny-sat-anomaly",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="LSTM-based anomaly detection for satellite telemetry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/tiny-sat-anomaly",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tiny-sat-train=src.train:main",
            "tiny-sat-eval=src.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
