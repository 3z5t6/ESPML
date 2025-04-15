from setuptools import setup, find_packages

setup(
    name="espml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0,<2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "flaml>=2.0.0",
        "joblib>=1.1.0",
        "requests",
        "psutil",
        "schedule",
    ],
)
