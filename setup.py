from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mental-wellness-model",
    version="0.1.0",
    author="EBD-Group-3",
    description="Big Data Engineering for Mental Wellness Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
    ],
)